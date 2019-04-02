#!/usr/bin/python3
from demo_utils.general import SUPPORTED_DATASETS
from IPython.display import Markdown as md
from demo_utils.get_hyper_params import get_hyper_params
from demo_utils.learning import get_model
from sklearn.model_selection import GridSearchCV
from demo_utils.general import get_data
import time
from demo_utils.general import gamest
from demo_utils.general import get_label
import json
import matplotlib.pyplot as plt
import numpy as np
# from matplotlib import rcParams


# Aquí va el código que tiene que ejecutar los tests, mirando el tiempo,
# y todos los útils que tiene


def get_prefixes(box_name):
    # Retorna el prefijo adecuado para acceder a parámetros del sampler y
    # de modelo, en función de la caja

    if box_name == 'none':
        sampler_prefix = 'sampler__'
        model_prefix = 'model__'
    elif box_name in ['grey_bag', 'grey_ens']:
        sampler_prefix = 'base_estimator__sampler__'
        model_prefix = 'base_estimator__model__'
    elif box_name in ['black_bag', 'black_ens']:
        sampler_prefix = 'sampler__'
        model_prefix = 'model__base_estimator__'

    return sampler_prefix, model_prefix


def cross_validate(model, tunning_params, data_train, target_train):
    clf = GridSearchCV(model, tunning_params, cv=10, iid=False)
    clf.fit(data_train, target_train)
    best_params = clf.best_params_
    return best_params


def exp(model_info, tunning_params, data_train, data_test, target_train,
        target_test, description='No description'):
    '''
    Ejecuta el experimento especificado y retorna un diccionario con los
    scores, los tiempos, label

    Esto NO es un experimento completo, solo es una columna de los experimentos
    '''

    model_name = model_info['model_name']
    if model_name in ['logit', 'linear_svc', 'rbf_svc']:
        param_name = 'C'
    elif model_name == 'dt':
        param_name = 'min_impurity_decrease'

    model = get_model(**model_info)
    box_name = model_info['box_type']
    sampler_name = model_info['sampler_name']
    sampler_prefix, model_prefix = get_prefixes(box_name)

    new_tunning_params = {}
    for k in tunning_params:
        new_k = model_prefix + \
            k if k in ['C', 'min_impurity_decrease'] else sampler_prefix + k
        new_tunning_params[new_k] = tunning_params[k]

    if sampler_name != 'identity':
        model.set_params(**{f'{sampler_prefix}n_components': 500})
    ##############################
    # Empieza el tiempo de ejecución
    ##############################
#     time0 = time.clock()
    time0 = time.perf_counter()

    if model_name == 'rbf_svc':
        chosen_gamma = gamest(data_train)
        model.set_params(**{f'{model_prefix}gamma': chosen_gamma})

    best_params = cross_validate(model=model,
                                 tunning_params=new_tunning_params,
                                 data_train=data_train,
                                 target_train=target_train)
    model.set_params(**best_params)

    model.fit(data_train, target_train)

#     time1 = time.clock()
    time1 = time.perf_counter()
    ##############################
    # Fin del tiempo de ejecución
    ##############################
    c_time = time1 - time0

    train_score = model.score(data_train, target_train)
    test_score = model.score(data_test, target_test)

    params_finales = model.get_params()
    model_param = {param_name: params_finales.get(
        model_prefix + param_name, 'Patata')}

    if model_name != 'rbf_svc':
        ret_gamma = params_finales.get(f'{sampler_prefix}gamma', None)
    else:
        ret_gamma = params_finales.get(f'{model_prefix}gamma', None)

    label = get_label(model_name=model_name,
                      sampler_name=sampler_name,
                      box_name=box_name,
                      n_estim=model_info['n_estim'])

    ret_dic = {
        'train_score': train_score,
        'test_score': test_score,
        'time': c_time,
        'model_param': model_param,
        # 'gamma': params_finales.get(f'{sampler_prefix}gamma', None),
        'gamma': ret_gamma,
        'label': label,
        'model_name': model_info['model_name'],
        'box_name': model_info['box_type'],
        'description': description,
    }
    print(ret_dic)
    return ret_dic


def exp1_1(dts_name):
    overfitting_gamma = 1000
    C_values = [10**i for i in range(4)]
    tunning_params = {'C': C_values}
    model1_info = {
        'model_name': 'rbf_svc',
        'model_params': {},
        'rbfsampler_gamma': None,
        'nystroem_gamma': None,
        'sampler_name': 'identity',
        'pca_bool': False,
        'pca_first': None,
        'n_estim': None,
        'box_type': 'none',
    }
    model2_info = {
        'model_name': 'linear_svc',
        'model_params': {},
        'rbfsampler_gamma': overfitting_gamma,
        'nystroem_gamma': None,
        'sampler_name': 'rbf',
        'pca_bool': False,
        'pca_first': None,
        'n_estim': None,
        'box_type': 'none',
    }
    model3_info = {
        'model_name': 'linear_svc',
        'model_params': {},
        'rbfsampler_gamma': None,
        'nystroem_gamma': overfitting_gamma,
        'sampler_name': 'nystroem',
        'pca_bool': False,
        'pca_first': None,
        'n_estim': None,
        'box_type': 'none',
    }

    data = get_data(dataset_name=dts_name, prop_train=2/3, n_ins=5000)
    data_train = data['data_train']
    data_test = data['data_test']
    target_train = data['target_train']
    target_test = data['target_test']

    d1 = exp(model_info=model1_info,
             tunning_params=tunning_params,
             data_train=data_train,
             data_test=data_test,
             target_train=target_train,
             target_test=target_test,
             description=f'A normal RBF-SVC with gamest ({dts_name})')
    d2 = exp(model_info=model2_info,
             tunning_params=tunning_params,
             data_train=data_train,
             data_test=data_test,
             target_train=target_train,
             target_test=target_test,
             description=f'A normal linear-SVC with RFF ({dts_name})')
    d3 = exp(model_info=model3_info,
             tunning_params=tunning_params,
             data_train=data_train,
             data_test=data_test,
             target_train=target_train,
             target_test=target_test,
             description=f'A normal linear-SVC with Nystroem ({dts_name})')

    store_exp(d1, d2, d3, exp_code='1_1', dts_name=dts_name)


def store_exp(*dics, exp_code, dts_name):
    '''
    exp_code : str
        Qué experimento se está realizando. De la forma 2_4
    dics : tuple of dict
        Diccionarios con los resultados de los experimentos
    '''
    filename = f'experimental_results/{exp_code}/{dts_name}.json'
    with open(filename, 'w') as f:
        json.dump(dics, f, indent=4, sort_keys=True)


def generate_png(exp_code, dts_name, labels, train_errors,
                 test_errors, times):

    mytimes = [-1 * i for i in times]
    filename = f'{exp_code}/{dts_name}'
    o_filename = f'experimental_graphs/{filename}.png'

    N = len(train_errors)
    ind = np.arange(N)
    width = 0.35

    fig, (x_errors, x_times) = plt.subplots(2, 1, sharex=True)
    fig.suptitle(dts_name)

    x_errors.bar(x=ind, height=test_errors, width=-width,
                 align='edge', label='Test', color='#10cc35')
    x_errors.bar(x=ind, height=train_errors,
                 width=width, align='edge', label='Train', color='#ef5045')

    x_errors.set_ylabel('Error')

    x_times.bar(x=ind, height=mytimes, width=2*width, color='#49dfed')
    x_times.set_ylabel('Time (s)')
    plt.xticks(ind, labels)

    ticks = x_times.get_yticks()
    x_times.set_yticklabels([int(abs(tick)) for tick in ticks])

    degree = 15
    x_errors.tick_params(rotation=degree)
    x_times.tick_params(rotation=degree)

    x_times.tick_params(axis='y', reset=True)
    x_errors.tick_params(axis='y', reset=True)

    x_errors.grid(True)
    x_times.grid(True)

    x_errors.legend(loc='best')

    plt.savefig(o_filename, bbox_inches="tight")


def generate_graphs(exp_code, dts_name):
    filename = f'{exp_code}/{dts_name}'
    i_filename = f'experimental_results/{filename}.json'
    o_filename = f'experimental_graphs/{filename}.png'

    with open(i_filename, 'r') as f:
        dic_list = json.load(f)

    train_scores = []
    test_scores = []
    labels = []
    times = []

    for e in dic_list:
        train_scores.append(e['train_score'])
        test_scores.append(e['test_score'])
        labels.append(e['label'])
        times.append(e['time'])

    train_errors = [1 - i for i in train_scores]
    test_errors = [1 - i for i in test_scores]

    generate_png(exp_code=exp_code, dts_name=dts_name, labels=labels,
                 train_errors=train_errors, test_errors=test_errors,
                 times=times)


def generate_graphs_exp1_1():
    exp_code = '1_1'
    for dts_name in SUPPORTED_DATASETS:
        generate_graphs(exp_code=exp_code, dts_name=dts_name)
