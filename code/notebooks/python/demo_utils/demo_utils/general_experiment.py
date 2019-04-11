#!/usr/bin/python3
from demo_utils.learning import get_model
import time
import json
from demo_utils.general import gamest
from demo_utils.general import get_label
from sklearn.model_selection import GridSearchCV


# Aquí está lo necesario para realizar un expeimento


def cross_validate(model, tunning_params, data_train, target_train):
    clf = GridSearchCV(model, tunning_params, cv=10, iid=False)
    clf.fit(data_train, target_train)
    best_params = clf.best_params_
    return best_params


def store_exp(*dics, exp_code, dts_name):
    '''
    Recibe diccionarios y los guarda en disco en json
    Parameters
    ==========
    dics : tuple of dict
        Diccionarios con los resultados de los experimentos
    exp_code : str
        Qué experimento se está realizando. De la forma 2_4
    '''
    filename = f'experimental_results/{exp_code}/{dts_name}.json'
    with open(filename, 'w') as f:
        json.dump(dics, f, indent=4, sort_keys=True)


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


def exp(model_info, tunning_params, data_train, data_test, target_train,
        target_test, description='No description'):
    '''
    Ejecuta el experimento especificado y retorna un diccionario con los
    scores, los tiempos, label

    Esto NO es un experimento completo, solo es una columna de los experimentos

    Es genérico, para cualquier experimento
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

    chosen_gamma = gamest(data_train)
    if model_name == 'rbf_svc':
        model.set_params(**{f'{model_prefix}gamma': chosen_gamma})
    elif sampler_name != 'identity':
        model.set_params(**{f'{sampler_prefix}gamma': chosen_gamma})

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


def store_exp_general(*dics, filename):
    '''
    Recibe diccionarios y los guarda en disco en json
    Parameters
    ==========
    dics : tuple of dict
        Diccionarios con los resultados de los experimentos
    exp_code : str
        Qué experimento se está realizando. De la forma 2_4
    '''
    # filename = f'experimental_results/{exp_code}/{dts_name}.json'
    with open(filename, 'w') as f:
        json.dump(dics, f, indent=4, sort_keys=True)
