from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix
import plotly.express as px
import time
import os
import implicit
from mab2rec import BanditRecommender, LearningPolicy
import pandas as pd
import numpy as np
from tqdm import tqdm

def load_data():
    df = pd.read_csv("/workspace/gregorio/reinforcement-learning-recsys/1-datasets/bestbuy/interactions.csv", sep=';')
    df = df.rename(columns={
        'id_user': 'user_id',
        'id_item': 'item_id',
    })
    df['response'] = 1
    df = df.sort_values(by='timestamp')
    df = df[['user_id', 'item_id', 'response']]
    df = df.iloc[:int(len(df) * 0.5)]
    return df

FACTORS = 10

def train_embeddings_model(Model, df, num_users, num_items, generate_embeddings=False):
    sparse_matrix = csr_matrix((df['response'], (df['user_id'], df['item_id'])), shape=(num_users, num_items))

    model = Model(factors=FACTORS, random_state=1)
    model.fit(sparse_matrix)

    if not generate_embeddings:
        return model, sparse_matrix
    
    # # Não precisamos mais do código abaixo, ele funcina para embeddings de usuário, não de itens
    # user_features_list = []

    # for user_id in df['user_id'].unique():
    #    user_factors = model.user_factors[user_id][:FACTORS]  # O BPR coloca 1 no final dos vetores latentes ?
    #    user_features_list.append([user_id] + list(user_factors))

    # df_user_features = pd.DataFrame(user_features_list, columns=['user_id'] + [f'u{i}' for i in range(FACTORS)])

    # model = model.to_cpu()
    return model, sparse_matrix, model.item_factors, model.user_factors

def test_embeddings_model(model, sparse_matrix, df_test):
    all_recs = []

    start_time = time.time()
    hits = 0
    for _, interaction in tqdm(df_test.iterrows(), total=len(df_test)):
        ids_recs, _ = model.recommend(userid=interaction['user_id'], user_items=sparse_matrix[interaction['user_id']], N=10)
        if interaction['item_id'] in ids_recs:
            hits += 1
        all_recs.append(ids_recs.tolist())
    
    recs_df = pd.DataFrame({
        'interaction_number': [i for i in range(len(df_test))],
        'user_id': df_test['user_id'],
        'item_id': df_test['item_id'],
        'recommendations': all_recs
    })
    
    return hits, hits/len(df_test), time.time() - start_time, recs_df

def train_mab(mab_algo, df_train_with_contexts, contexts_col):
    contexts = get_concat_context(df_train_with_contexts, contexts_col)
    mab_algo.fit(
        decisions=df_train_with_contexts['item_id'],
        rewards=df_train_with_contexts['response'],
        contexts=contexts
    )


def test_non_incremental(mab_algo, contexts_col, df_test, interactions_by_user):
    start_time = time.time()
    hits = 0

    # contexts = df_test.merge(user_features, how='left', on='user_id').drop(columns=['user_id', 'item_id', 'response']).values
    # contexts = np.array(df_test[contexts_col].tolist())
    print('entrou')
    contexts = get_concat_context(df_test, contexts_col)
    filters = df_test.merge(interactions_by_user, how='left', on='user_id')[['interactions']].values.squeeze(axis=1) 
    print('saiu')

    recomendations = mab_algo.recommend(contexts, filters, apply_sigmoid=False)

    df_test = df_test.reset_index(drop=True)

    hits = 0
    for i, interaction in tqdm(df_test.iterrows(), total=len(df_test)):
        if interaction['item_id'] in recomendations[i]:
            hits += 1
    

    recs_df = pd.DataFrame({
        'interaction_number': [i for i in range(len(df_test))],
        'user_id': df_test['user_id'],
        'item_id': df_test['item_id'],
        'recommendations': recomendations
    })

    return hits, hits/len(df_test), time.time() - start_time, recs_df

def group_interactions_by_user(interactions_df):
    interactions_by_user = interactions_df\
                        .groupby('user_id')[['item_id']]\
                        .apply(lambda df_user: df_user['item_id'].tolist())\
                        .reset_index(name='interactions')
    interactions_by_user = interactions_by_user.reset_index(drop=True)
    return interactions_by_user

def create_contexts_list_items_mean(interactions_df, items_embeddings):
    users_current_info = {}
    contexts = []

    for _, row in tqdm(interactions_df.iterrows(), total=len(interactions_df)):
        user_id = row["user_id"]
        item_id = row["item_id"]

        if user_id not in users_current_info:
            users_current_info[user_id] = {
                'acum_emb': np.zeros((FACTORS, )),
                'count': 0
            }
        
        contexts.append(users_current_info[user_id]['acum_emb'] / max(1, users_current_info[user_id]['count']))

        users_current_info[user_id]['acum_emb'] += items_embeddings[item_id][:FACTORS]
        users_current_info[user_id]['count'] += 1

    return contexts

def create_contexts_list_items_concat(interactions_df, items_embeddings, window_size):
    users_current_info = {}
    contexts = []

    for _, row in interactions_df.iterrows():
        user_id = row["user_id"]
        item_id = row["item_id"]

        if user_id not in users_current_info:
            users_current_info[user_id] = np.zeros((window_size, FACTORS))
        
        contexts.append(users_current_info[user_id].flatten())
        
        users_current_info[user_id][1:] = users_current_info[user_id][:-1]
        users_current_info[user_id][0] = items_embeddings[item_id][:FACTORS]

    return contexts

def create_contexts_list_user(interactions_df, users_embeddings):
    contexts = []

    for _, row in interactions_df.iterrows():
        user_id = row["user_id"]
        contexts.append(users_embeddings[user_id][:FACTORS])

    return contexts

def get_concat_context(interactions, context_cols):
    # Concat multiple array columns into a single array column
    return np.array(interactions[context_cols].apply(lambda x: np.concatenate((*x, [1])), axis=1).tolist())  # MUDANÇA: adiciona 1 ao final de cada vetor (bias)


def test(test_size, train_initial_size, train_extra_increment_step_size, windows_sizes):
    '''
    - `test_size`: define o tamanho da partição de teste no train/test split inicial. Por exemplo, se for escolhido 0.1 (10%), a partição de teste terá 10% das interações e a partição de treino terá 90% das interações. O tamanho da partição de teste passará ainda por um filtro com o tamanho do treino inicial, definido no próximo parâmetro.
    - `train_initial_size`: define o tamanho inicial que será usado para treino dos modelos. Esse tamanho é uma porcentagem da partição de treino, por exemplo, 0.5 (50%) quer dizer que o treino será feito inicialmente com 50% das interações separadas para treino. Vale ressaltar que essa porcentagem é relacionada apenas à partição de treino, então, se temos uma partição de treino de 0.9 (90%) e o “train_initial_size” é definido como 0.5 (50%), então, teremos 45% (0.9 * 0.5) das interações todas para o treino inicial. Com a base de treino separada com essa porcentagem inicial, a base de teste passara por um filtro, removendo todas as interações com itens ou usuários que nunca foram vistos nesse treino inicial.
    - `train_extra_increment_step_size`: define a porcentagem do "treinamento extra" que será usado. No início a base de dados é separada em treino inicial (train_initial_size), "treinamento extra" e teste. O "treinamento extra", assim como o teste, passa por um filtro para remover interações com itens ou usuários que nunca foram vistos no treino inicial. Após o treino inicial, o "treinamento extra" é usado para treinar os modelos de embeddings e os modelos de bandit. O "treinamento extra" é incrementado a cada iteração, de acordo com o valor desse parâmetro. Por exemplo, se o `train_extra_increment_step_size` é 0.1 (10%), então, a cada iteração, 10% das interações são adicionadas ao treino, até que todo o "treinamento extra" seja usado.
    - `windows_sizes`: tamanho das janelas de contextos que serão usadas para teste. Por exemplo, se for passado [3, 5, 7], as janelas de tamanho de 3, 5 e 7 serão usadas como contexto para treinar os modelos de MAB (gerando resultados diferentes para cada tamanho de janela).
    '''
    results = []
    df_recs = pd.DataFrame(columns=['algorithm', 'interaction_number', 'user_id', 'item_id', 'recommendations'])
    # df_train = pd.read_csv(train_data)
    # df_test = pd.read_csv(test_data)

    df_full = load_data()

    df_full['user_id'] = LabelEncoder().fit_transform(df_full['user_id'])
    df_full['item_id'] = LabelEncoder().fit_transform(df_full['item_id'])

    num_users = df_full['user_id'].nunique()
    num_items = df_full['item_id'].nunique()

    split_index = int(len(df_full) * (1 - test_size))
    df_train_full = df_full[:split_index]
    df_test = df_full[split_index:]

    initial_df_train = df_train_full[:int(len(df_train_full) * train_initial_size)]
    extra_df_train = df_train_full[int(len(df_train_full) * train_initial_size):]
    extra_df_train = extra_df_train[(extra_df_train['user_id'].isin(initial_df_train['user_id'])) & (extra_df_train['item_id'].isin(initial_df_train['item_id']))]
    extra_df_train = extra_df_train.reset_index(drop=True)

    df_test = df_test[(df_test['user_id'].isin(initial_df_train['user_id'])) & (df_test['item_id'].isin(initial_df_train['item_id']))]
    df_test = df_test.reset_index(drop=True)
    df_test_for_evaluation = df_test[df_test['response'] == 1]
    df_test_for_evaluation = df_test_for_evaluation.reset_index(drop=True)

    print('Generating ALS embeddings')
    ALS_model, _, ALS_item_embeddings, ALS_user_embeddings = train_embeddings_model(implicit.als.AlternatingLeastSquares, initial_df_train, num_users, num_items, generate_embeddings=True)

    print('Generating BPR embeddings')
    BPR_model, _, BPR_item_embeddings, BPR_user_embeddings = train_embeddings_model(implicit.bpr.BayesianPersonalizedRanking, initial_df_train, num_users, num_items, generate_embeddings=True)

    '''
    for window_size in windows_sizes:
        print(f'Generating contexts for window size of {window_size} (contat items emb)')
        df_full_new = pd.concat([initial_df_train, extra_df_train, df_test_for_evaluation])
        als_contexts = create_contexts_list_items_concat(df_full_new, ALS_item_embeddings, window_size)
        bpr_contexts = create_contexts_list_items_concat(df_full_new, BPR_item_embeddings, window_size)

        initial_df_train[f'als_context_item_concat_{window_size}'] = als_contexts[:len(initial_df_train)]
        initial_df_train[f'bpr_context_item_concat_{window_size}'] = bpr_contexts[:len(initial_df_train)]

        extra_df_train[f'als_context_item_concat_{window_size}'] = als_contexts[len(initial_df_train):len(initial_df_train) + len(extra_df_train)]
        extra_df_train[f'bpr_context_item_concat_{window_size}'] = bpr_contexts[len(initial_df_train):len(initial_df_train) + len(extra_df_train)]

        df_test_for_evaluation[f'als_context_item_concat_{window_size}'] = als_contexts[len(initial_df_train) + len(extra_df_train):]
        df_test_for_evaluation[f'bpr_context_item_concat_{window_size}'] = bpr_contexts[len(initial_df_train) + len(extra_df_train):]

    print('Generating contexts for user embeddings')
    df_full_new = pd.concat([initial_df_train, extra_df_train, df_test_for_evaluation])
    als_contexts = create_contexts_list_user(df_full_new, ALS_user_embeddings)
    bpr_contexts = create_contexts_list_user(df_full_new, BPR_user_embeddings)

    initial_df_train['als_context_user'] = als_contexts[:len(initial_df_train)]
    initial_df_train['bpr_context_user'] = bpr_contexts[:len(initial_df_train)]

    extra_df_train['als_context_user'] = als_contexts[len(initial_df_train):len(initial_df_train) + len(extra_df_train)]
    extra_df_train['bpr_context_user'] = bpr_contexts[len(initial_df_train):len(initial_df_train) + len(extra_df_train)]

    df_test_for_evaluation['als_context_user'] = als_contexts[len(initial_df_train) + len(extra_df_train):]
    df_test_for_evaluation['bpr_context_user'] = bpr_contexts[len(initial_df_train) + len(extra_df_train):]
    '''
    
    print('Generating contexts for item mean embeddings')
    df_full_new = pd.concat([initial_df_train, extra_df_train, df_test_for_evaluation])
    als_contexts = create_contexts_list_items_mean(df_full_new, ALS_item_embeddings)
    bpr_contexts = create_contexts_list_items_mean(df_full_new, BPR_item_embeddings)

    initial_df_train['als_context_items_mean'] = als_contexts[:len(initial_df_train)]
    initial_df_train['bpr_context_items_mean'] = bpr_contexts[:len(initial_df_train)]

    extra_df_train['als_context_items_mean'] = als_contexts[len(initial_df_train):len(initial_df_train) + len(extra_df_train)]
    extra_df_train['bpr_context_items_mean'] = bpr_contexts[len(initial_df_train):len(initial_df_train) + len(extra_df_train)]

    df_test_for_evaluation['als_context_items_mean'] = als_contexts[len(initial_df_train) + len(extra_df_train):]
    df_test_for_evaluation['bpr_context_items_mean'] = bpr_contexts[len(initial_df_train) + len(extra_df_train):]

    algos_dict = {
        # 'item_concat': {
        #     'item_concat': True,
        #     'item_mean': False,
        #     'user': False
        # },
        'item_mean': {
            'item_concat': False,
            'item_mean': True,
            'user': False
        },
        # 'user': {
        #     'item_concat': False,
        #     'item_mean': False,
        #     'user': True
        # },
        # 'item_concat-item_mean': {
        #     'item_concat': True,
        #     'item_mean': True,
        #     'user': False
        # },
        # 'item_concat-user': {
        #     'item_concat': True,
        #     'item_mean': False,
        #     'user': True
        # },
        # 'item_mean-user': {
        #     'item_concat': False,
        #     'item_mean': True,
        #     'user': True
        # },
        # 'all': {
        #     'item_concat': True,
        #     'item_mean': True,
        #     'user': True
        # }
    }

    for algo_name, _ in algos_dict.items():
        algos_dict[algo_name]['results'] = []
        algos_dict[algo_name]['df_recs'] = pd.DataFrame(columns=['algorithm', 'interaction_number', 'user_id', 'item_id', 'recommendations'])

    def save_algo_result(algo_name, hits, hr, spent_time, df_recs_algo, current_extra_train_size, results):
        df_recs_algo['algorithm'] = algo_name
        df_recs_algo['train_size'] = current_extra_train_size
        df_recs_new = pd.concat([df_recs, df_recs_algo])
        results.append({'algorithm': algo_name, 'hits': hits, 'hr': hr, 'time': spent_time, 'train_size': current_extra_train_size})
        return df_recs_new

    current_extra_train_size = 0
    while current_extra_train_size <= 1:
        print(f"Current extra train size: {current_extra_train_size}")

        current_df_train = pd.concat([initial_df_train, extra_df_train[:int(len(extra_df_train) * current_extra_train_size)]])
        interactions_by_user = group_interactions_by_user(current_df_train)  # MUDANÇA AQUI

        # -------------- ALS -----------------
        print('Training ALS')
        ALS_model, sparse_matrix = train_embeddings_model(implicit.als.AlternatingLeastSquares, current_df_train, num_users, num_items)

        print('Testing ALS')
        hits, hr, spent_time, df_recs_als = test_embeddings_model(ALS_model, sparse_matrix, df_test_for_evaluation)
        df_recs = save_algo_result('ALS', hits, hr, spent_time, df_recs_als, current_extra_train_size, results)

        # -------------- BPR -----------------
        print('Training BPR')
        BPR_model, sparse_matrix = train_embeddings_model(implicit.bpr.BayesianPersonalizedRanking, current_df_train, num_users, num_items)

        print('Testing BPR')
        hits, hr, spent_time, df_recs_bpr = test_embeddings_model(BPR_model, sparse_matrix, df_test_for_evaluation)
        df_recs = save_algo_result('BPR', hits, hr, spent_time, df_recs_bpr, current_extra_train_size, results)
        
        for algo_name, algo_dict in algos_dict.items():
            if algo_dict['item_concat']:
                windows = windows_sizes
            else:
                windows = [None]
            
            for window_size in windows:
                als_embeddings_cols = []
                bpr_embeddings_cols = []
                print_extra = f' - {algo_name}'
                algo_name_extra = ''
                if algo_dict['item_concat']:
                    als_embeddings_cols.append(f'als_context_item_concat_{window_size}')
                    bpr_embeddings_cols.append(f'bpr_context_item_concat_{window_size}')
                    print_extra = f' - {algo_name} - {window_size}'
                    algo_name_extra = f' - {window_size}'
                if algo_dict['item_mean']:
                    als_embeddings_cols.append('als_context_items_mean')
                    bpr_embeddings_cols.append('bpr_context_items_mean')
                if algo_dict['user']:
                    als_embeddings_cols.append('als_context_user')
                    bpr_embeddings_cols.append('bpr_context_user')
                
                # ------ LinUCB - ALS embeddings -------
                print(f'Training LinUCB - ALS embeddings{print_extra}')
                linUCB_model = BanditRecommender(learning_policy=LearningPolicy.LinUCB(alpha=0.1), top_k=10)
                start_time = time.time()
                train_mab(linUCB_model, current_df_train, als_embeddings_cols)  # Mudança no treinamento dos MAB
                print(f'Treinamento demorou {time.time() - start_time} segundos')

                print(f'Testing LinUCB - ALS embeddings{print_extra}')
                hits, hr, spent_time, df_recs_linUCB = test_non_incremental(linUCB_model, als_embeddings_cols, df_test_for_evaluation, interactions_by_user)
                algo_dict['df_recs'] = save_algo_result(f'LinUCB - ALS embeddings{algo_name_extra}', hits, hr, spent_time, df_recs_linUCB, current_extra_train_size, algo_dict['results'])


                # ------ LinUCB - BPR embeddings -------
                print(f'Training LinUCB - BPR embeddings{print_extra}')
                linUCB_model = BanditRecommender(learning_policy=LearningPolicy.LinUCB(alpha=0.1), top_k=10)
                train_mab(linUCB_model, current_df_train, bpr_embeddings_cols)

                print(f'Testing LinUCB - BPR embeddings{print_extra}')
                hits, hr, spent_time, df_recs_linUCB = test_non_incremental(linUCB_model, bpr_embeddings_cols, df_test_for_evaluation, interactions_by_user)
                algo_dict['df_recs'] = save_algo_result(f'LinUCB - BPR embeddings{algo_name_extra}', hits, hr, spent_time, df_recs_linUCB, current_extra_train_size, algo_dict['results'])

                # ------ LinGreedy - ALS embeddings -------
                print(f'Training LinGreedy - ALS embeddings{print_extra}')
                linGreedy_model = BanditRecommender(learning_policy=LearningPolicy.LinGreedy(epsilon=0.01), top_k=10)
                train_mab(linGreedy_model, current_df_train, als_embeddings_cols)

                print(f'Testing LinGreedy - ALS embeddings{print_extra}')
                hits, hr, spent_time, df_recs_linGreedy = test_non_incremental(linGreedy_model, als_embeddings_cols, df_test_for_evaluation, interactions_by_user)
                algo_dict['df_recs'] = save_algo_result(f'LinGreedy - ALS embeddings{algo_name_extra}', hits, hr, spent_time, df_recs_linGreedy, current_extra_train_size, algo_dict['results'])


                # ------ LinGreedy - BPR embeddings -------
                print(f'Training LinGreedy - BPR embeddings{print_extra}')
                linGreedy_model = BanditRecommender(learning_policy=LearningPolicy.LinGreedy(epsilon=0.01), top_k=10)
                train_mab(linGreedy_model, current_df_train, bpr_embeddings_cols)

                print(f'Testing LinGreedy - BPR embeddings{print_extra}')
                hits, hr, spent_time, df_recs_linGreedy = test_non_incremental(linGreedy_model, bpr_embeddings_cols, df_test_for_evaluation, interactions_by_user)
                algo_dict['df_recs'] = save_algo_result(f'LinGreedy - BPR embeddings{algo_name_extra}', hits, hr, spent_time, df_recs_linGreedy, current_extra_train_size, algo_dict['results'])
        
        # Incrementando o tamanho do treino para próxima iteração
        current_extra_train_size = round(current_extra_train_size + train_extra_increment_step_size, 2)
    
    save_path = f'results-v14/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    df_results_als_bpr = pd.DataFrame(results)
    for algo_name, algo_dict in algos_dict.items():
        df_results_final = pd.DataFrame(algo_dict['results'])
        df_results_final = pd.concat([df_results_final, df_results_als_bpr])
        df_results_final = df_results_final.astype({'hits': int, 'hr': float, 'time': float})
        df_results_final['test_size'] = round(test_size, 2)
        df_results_final['test_interactions'] = len(df_test_for_evaluation)

        df_results_final.to_csv(f'{save_path}/results-{algo_name}.csv', index=False)

        df_recs_final = pd.concat([df_recs, algo_dict['df_recs']])
        df_recs_final.to_csv(f'{save_path}/recs-{algo_name}.csv', index=False)


test(test_size=0.1, train_initial_size=0.5, train_extra_increment_step_size=0.1, windows_sizes=[1,2,3,4,5])

df_results = pd.read_csv('results-v14/results-item_mean.csv')

def remove_incremental(df_results):
    new_df = df_results[(~df_results['algorithm'].str.contains('incremental') | df_results['algorithm'].str.contains('non-incremental'))]
    new_df['algorithm'] = new_df['algorithm'].str.replace(' - non-incremental', '')
    return new_df

def transform_in_upper_and_lower_bounds(df_results, algo_names):
    train_sizes = sorted(df_results['train_size'].unique().tolist())
    min_train_size = train_sizes[0]
    max_train_size = train_sizes[-1]
    qnt_train_sizes = len(train_sizes)

    for algo_name in algo_names:
        algo_row_lower = df_results[(df_results['algorithm'] == algo_name) & (df_results['train_size'] == min_train_size)]
        df_lower = pd.DataFrame({
            'algorithm': [f'{algo_name} lower'] * qnt_train_sizes,
            'hits': [algo_row_lower['hits'].values[0]] * qnt_train_sizes,
            'hr': [algo_row_lower['hr'].values[0]] * qnt_train_sizes,
            'time': [algo_row_lower['time'].values[0]] * qnt_train_sizes,
            'train_size': train_sizes,
            'test_size': [algo_row_lower['test_size'].values[0]] * qnt_train_sizes,
            'test_interactions': [algo_row_lower['test_interactions'].values[0]] * qnt_train_sizes
        })
        df_results = pd.concat([df_results, df_lower])

        algo_row_upper = df_results[(df_results['algorithm'] == algo_name) & (df_results['train_size'] == max_train_size)]
        df_upper = pd.DataFrame({
            'algorithm': [f'{algo_name} upper'] * qnt_train_sizes,
            'hits': [algo_row_upper['hits'].values[0]] * qnt_train_sizes,
            'hr': [algo_row_upper['hr'].values[0]] * qnt_train_sizes,
            'time': [algo_row_upper['time'].values[0]] * qnt_train_sizes,
            'train_size': train_sizes,
            'test_size': [algo_row_upper['test_size'].values[0]] * qnt_train_sizes,
            'test_interactions': [algo_row_upper['test_interactions'].values[0]] * qnt_train_sizes
        })
        df_results = pd.concat([df_results, df_upper])

        df_results = df_results[df_results['algorithm'] != algo_name]
    
    return df_results

import plotly.graph_objects as go

def plot_results(df_results, save_root):
    df_results = remove_incremental(df_results)
    df_results = transform_in_upper_and_lower_bounds(df_results, ['ALS', 'BPR'])

    algos_configs = {
        'ALS upper': {'color': 'blue', 'dash': 'dash'},
        'ALS lower': {'color': 'blue', 'dash': 'dash'},
        'BPR upper': {'color': 'red', 'dash': 'dash'},
        'BPR lower': {'color': 'red', 'dash': 'dash'},
        'LinUCB - ALS embeddings': {'color': 'green', 'dash': 'solid'},
        'LinUCB - BPR embeddings': {'color': 'purple', 'dash': 'solid'},
        'LinGreedy - ALS embeddings': {'color': 'orange', 'dash': 'solid'},
        'LinGreedy - BPR embeddings': {'color': 'pink', 'dash': 'solid'}
    }

    fig = go.Figure()
    for algo_name, config in algos_configs.items():
        df_algo = df_results[df_results['algorithm'] == algo_name]
        fig.add_trace(go.Scatter(x=df_algo['train_size'], y=df_algo['hr'], mode='lines', name=algo_name, line=dict(color=config['color'], dash=config['dash'])))
    
    fig.update_layout(title='HR x Train size', xaxis_title='Train size', yaxis_title='HR')
    #  fig.show()

    fig.write_html(f'{save_root}/hr_x_train_size.html')
    fig.write_image(f'{save_root}/hr_x_train_size.png')

plot_results(df_results, 'results-v14')