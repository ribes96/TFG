#!/usr/bin/python3

# from demo_utils.testing_code import store_exp
from demo_utils.general_experiment import exp
from demo_utils.general_experiment import store_exp
from demo_utils.general import get_data
import json


def exp1_1(dts_name):
    exp_code = '1_1'
    # overfitting_gamma = 1000
    # C_values = [10**i for i in range(4)]
    C_values = [0.5, 1, 5, 20, 50]
    tunning_params = {'C': C_values}
    box_type = 'none'
    model1_info = {
        'model_name': 'rbf_svc',
        'model_params': {},
        'rbfsampler_gamma': None,
        'nystroem_gamma': None,
        'sampler_name': 'identity',
        'pca_bool': False,
        'pca_first': None,
        'n_estim': None,
        'box_type': box_type,
    }
    model2_info = {
        'model_name': 'linear_svc',
        'model_params': {},
        'rbfsampler_gamma': None,
        'nystroem_gamma': None,
        'sampler_name': 'rbf',
        'pca_bool': False,
        'pca_first': None,
        'n_estim': None,
        'box_type': box_type,
    }
    model3_info = {
        'model_name': 'linear_svc',
        'model_params': {},
        'rbfsampler_gamma': None,
        'nystroem_gamma': None,
        'sampler_name': 'nystroem',
        'pca_bool': False,
        'pca_first': None,
        'n_estim': None,
        'box_type': box_type,
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

    store_exp(d1, d2, d3, exp_code=exp_code, dts_name=dts_name)


def exp2_1(dts_name):
    exp_code = '2_1'
    model_name = 'logit'
    C_value = {'C': 1000}
    box_type = 'none'
    n_estim = None
    # C_values = [10**i for i in range(4)]
    # tunning_params = {'C': C_values}
    tunning_params = {}
    model1_info = {
        'model_name': model_name,
        'model_params': C_value,
        'rbfsampler_gamma': None,
        'nystroem_gamma': None,
        'sampler_name': 'identity',
        'pca_bool': False,
        'pca_first': None,
        'n_estim': n_estim,
        'box_type': box_type,
    }
    model2_info = {
        'model_name': model_name,
        'model_params': C_value,
        'rbfsampler_gamma': None,
        'nystroem_gamma': None,
        'sampler_name': 'rbf',
        'pca_bool': False,
        'pca_first': None,
        'n_estim': n_estim,
        'box_type': box_type,
    }
    model3_info = {
        'model_name': model_name,
        'model_params': C_value,
        'rbfsampler_gamma': None,
        'nystroem_gamma': None,
        'sampler_name': 'nystroem',
        'pca_bool': False,
        'pca_first': None,
        'n_estim': n_estim,
        'box_type': box_type,
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
             description=f'Normal logit without regularization ({dts_name})')
    d2 = exp(
        model_info=model2_info,
        tunning_params=tunning_params,
        data_train=data_train,
        data_test=data_test,
        target_train=target_train,
        target_test=target_test,
        description=f'Logit with RFF without regularization ({dts_name})')
    d3 = exp(
        model_info=model3_info,
        tunning_params=tunning_params,
        data_train=data_train,
        data_test=data_test,
        target_train=target_train,
        target_test=target_test,
        description=f'Logit with Nystroem without regularization ({dts_name})')

    store_exp(d1, d2, d3, exp_code=exp_code, dts_name=dts_name)


def exp2_2(dts_name):
    exp_code = '2_2'
    model_name = 'logit'
    C_value = {'C': 1000}
    box_type = 'black_bag'
    n_estim = 50
    # C_values = [10**i for i in range(4)]
    # tunning_params = {'C': C_values}
    tunning_params = {}
    model1_info = {
        'model_name': model_name,
        'model_params': C_value,
        'rbfsampler_gamma': None,
        'nystroem_gamma': None,
        'sampler_name': 'identity',
        'pca_bool': False,
        'pca_first': None,
        'n_estim': None,
        'box_type': 'none',
    }
    model2_info = {
        'model_name': model_name,
        'model_params': C_value,
        'rbfsampler_gamma': None,
        'nystroem_gamma': None,
        'sampler_name': 'rbf',
        'pca_bool': False,
        'pca_first': None,
        'n_estim': n_estim,
        'box_type': box_type,
    }
    model3_info = {
        'model_name': model_name,
        'model_params': C_value,
        'rbfsampler_gamma': None,
        'nystroem_gamma': None,
        'sampler_name': 'nystroem',
        'pca_bool': False,
        'pca_first': None,
        'n_estim': n_estim,
        'box_type': box_type,
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
             description=f'Normal logit without regularization ({dts_name})')
    d2 = exp(
        model_info=model2_info,
        tunning_params=tunning_params,
        data_train=data_train,
        data_test=data_test,
        target_train=target_train,
        target_test=target_test,
        description=f'Logit black_bag with RFF without regul. ({dts_name})')
    d3 = exp(
        model_info=model3_info,
        tunning_params=tunning_params,
        data_train=data_train,
        data_test=data_test,
        target_train=target_train,
        target_test=target_test,
        description=f'Logit black_bag with Nys. without regul. ({dts_name})')

    store_exp(d1, d2, d3, exp_code=exp_code, dts_name=dts_name)


def exp2_3(dts_name):
    exp_code = '2_3'
    model_name = 'logit'
    C_value = {'C': 1000}
    box_type = 'grey_bag'
    n_estim = 50
    # C_values = [10**i for i in range(4)]
    # tunning_params = {'C': C_values}
    tunning_params = {}
    model1_info = {
        'model_name': model_name,
        'model_params': C_value,
        'rbfsampler_gamma': None,
        'nystroem_gamma': None,
        'sampler_name': 'identity',
        'pca_bool': False,
        'pca_first': None,
        'n_estim': None,
        'box_type': 'none',
    }
    model2_info = {
        'model_name': model_name,
        'model_params': C_value,
        'rbfsampler_gamma': None,
        'nystroem_gamma': None,
        'sampler_name': 'rbf',
        'pca_bool': False,
        'pca_first': None,
        'n_estim': n_estim,
        'box_type': box_type,
    }
    model3_info = {
        'model_name': model_name,
        'model_params': C_value,
        'rbfsampler_gamma': None,
        'nystroem_gamma': None,
        'sampler_name': 'nystroem',
        'pca_bool': False,
        'pca_first': None,
        'n_estim': n_estim,
        'box_type': box_type,
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
             description=f'Normal logit without regularization ({dts_name})')
    d2 = exp(
        model_info=model2_info,
        tunning_params=tunning_params,
        data_train=data_train,
        data_test=data_test,
        target_train=target_train,
        target_test=target_test,
        description=f'Logit grey_bag with RFF without regul. ({dts_name})')
    d3 = exp(
        model_info=model3_info,
        tunning_params=tunning_params,
        data_train=data_train,
        data_test=data_test,
        target_train=target_train,
        target_test=target_test,
        description=f'Logit grey_bag with Nys. without regul. ({dts_name})')

    store_exp(d1, d2, d3, exp_code=exp_code, dts_name=dts_name)


def exp2_4(dts_name):
    exp_code = '2_4'
    model_name = 'logit'
    C_value = {'C': 1000}
    box_type = 'grey_ens'
    n_estim = 50
    # C_values = [10**i for i in range(4)]
    # tunning_params = {'C': C_values}
    tunning_params = {}
    model1_info = {
        'model_name': model_name,
        'model_params': C_value,
        'rbfsampler_gamma': None,
        'nystroem_gamma': None,
        'sampler_name': 'identity',
        'pca_bool': False,
        'pca_first': None,
        'n_estim': None,
        'box_type': 'none',
    }
    model2_info = {
        'model_name': model_name,
        'model_params': C_value,
        'rbfsampler_gamma': None,
        'nystroem_gamma': None,
        'sampler_name': 'rbf',
        'pca_bool': False,
        'pca_first': None,
        'n_estim': n_estim,
        'box_type': box_type,
    }
    model3_info = {
        'model_name': model_name,
        'model_params': C_value,
        'rbfsampler_gamma': None,
        'nystroem_gamma': None,
        'sampler_name': 'nystroem',
        'pca_bool': False,
        'pca_first': None,
        'n_estim': n_estim,
        'box_type': box_type,
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
             description=f'Normal logit without regularization ({dts_name})')
    d2 = exp(
        model_info=model2_info,
        tunning_params=tunning_params,
        data_train=data_train,
        data_test=data_test,
        target_train=target_train,
        target_test=target_test,
        description=f'Logit grey_ens with RFF without regul. ({dts_name})')
    d3 = exp(
        model_info=model3_info,
        tunning_params=tunning_params,
        data_train=data_train,
        data_test=data_test,
        target_train=target_train,
        target_test=target_test,
        description=f'Logit grey_ens with Nys. without regul. ({dts_name})')

    store_exp(d1, d2, d3, exp_code=exp_code, dts_name=dts_name)


def exp2_5(dts_name):
    exp_code = '2_5'
    model_name = 'linear_svc'
    # C_value = {'C': 1000}
    box_type = 'none'
    n_estim = None
    C_values = [10**i for i in range(4)]
    tunning_params = {'C': C_values}
    # tunning_params = {}
    model1_info = {
        'model_name': model_name,
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
        'model_name': model_name,
        'model_params': {},
        'rbfsampler_gamma': None,
        'nystroem_gamma': None,
        'sampler_name': 'rbf',
        'pca_bool': False,
        'pca_first': None,
        'n_estim': n_estim,
        'box_type': box_type,
    }
    model3_info = {
        'model_name': model_name,
        'model_params': {},
        'rbfsampler_gamma': None,
        'nystroem_gamma': None,
        'sampler_name': 'nystroem',
        'pca_bool': False,
        'pca_first': None,
        'n_estim': n_estim,
        'box_type': box_type,
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
             description=f'Linear SVM ({dts_name})')
    d2 = exp(
        model_info=model2_info,
        tunning_params=tunning_params,
        data_train=data_train,
        data_test=data_test,
        target_train=target_train,
        target_test=target_test,
        description=f'Linear SVM with RFF ({dts_name})')
    d3 = exp(
        model_info=model3_info,
        tunning_params=tunning_params,
        data_train=data_train,
        data_test=data_test,
        target_train=target_train,
        target_test=target_test,
        description=f'Linear SVM with Nystroem ({dts_name})')

    store_exp(d1, d2, d3, exp_code=exp_code, dts_name=dts_name)


def exp2_6(dts_name):
    exp_code = '2_6'
    model_name = 'linear_svc'
    C_value = {'C': 1000}
    box_type = 'black_bag'
    n_estim = 50
    C_values = [10**i for i in range(4)]
    tunning_params = {'C': C_values}
    # tunning_params = {}
    model1_info = {
        'model_name': model_name,
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
        'model_name': model_name,
        'model_params': C_value,
        'rbfsampler_gamma': None,
        'nystroem_gamma': None,
        'sampler_name': 'rbf',
        'pca_bool': False,
        'pca_first': None,
        'n_estim': n_estim,
        'box_type': box_type,
    }
    model3_info = {
        'model_name': model_name,
        'model_params': C_value,
        'rbfsampler_gamma': None,
        'nystroem_gamma': None,
        'sampler_name': 'nystroem',
        'pca_bool': False,
        'pca_first': None,
        'n_estim': n_estim,
        'box_type': box_type,
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
             description=f'Linear SVM ({dts_name})')
    d2 = exp(
        model_info=model2_info,
        tunning_params={},
        data_train=data_train,
        data_test=data_test,
        target_train=target_train,
        target_test=target_test,
        description=f'Linear SVM with RFF Black Bag ({dts_name})')
    d3 = exp(
        model_info=model3_info,
        tunning_params={},
        data_train=data_train,
        data_test=data_test,
        target_train=target_train,
        target_test=target_test,
        description=f'Linear SVM with Nystroem Black Bag ({dts_name})')

    store_exp(d1, d2, d3, exp_code=exp_code, dts_name=dts_name)


def exp2_7(dts_name):
    exp_code = '2_7'
    model_name = 'linear_svc'
    C_value = {'C': 1000}
    box_type = 'grey_bag'
    n_estim = 50
    C_values = [10**i for i in range(4)]
    tunning_params = {'C': C_values}
    # tunning_params = {}
    model1_info = {
        'model_name': model_name,
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
        'model_name': model_name,
        'model_params': C_value,
        'rbfsampler_gamma': None,
        'nystroem_gamma': None,
        'sampler_name': 'rbf',
        'pca_bool': False,
        'pca_first': None,
        'n_estim': n_estim,
        'box_type': box_type,
    }
    model3_info = {
        'model_name': model_name,
        'model_params': C_value,
        'rbfsampler_gamma': None,
        'nystroem_gamma': None,
        'sampler_name': 'nystroem',
        'pca_bool': False,
        'pca_first': None,
        'n_estim': n_estim,
        'box_type': box_type,
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
             description=f'Linear SVM ({dts_name})')
    d2 = exp(
        model_info=model2_info,
        tunning_params={},
        data_train=data_train,
        data_test=data_test,
        target_train=target_train,
        target_test=target_test,
        description=f'Linear SVM with RFF Black Bag ({dts_name})')
    d3 = exp(
        model_info=model3_info,
        tunning_params={},
        data_train=data_train,
        data_test=data_test,
        target_train=target_train,
        target_test=target_test,
        description=f'Linear SVM with Nystroem Black Bag ({dts_name})')

    store_exp(d1, d2, d3, exp_code=exp_code, dts_name=dts_name)


def exp2_8(dts_name):
    exp_code = '2_8'
    model_name = 'linear_svc'
    C_value = {'C': 1000}
    box_type = 'grey_ens'
    n_estim = 50
    C_values = [10**i for i in range(4)]
    tunning_params = {'C': C_values}
    # tunning_params = {}
    model1_info = {
        'model_name': model_name,
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
        'model_name': model_name,
        'model_params': C_value,
        'rbfsampler_gamma': None,
        'nystroem_gamma': None,
        'sampler_name': 'rbf',
        'pca_bool': False,
        'pca_first': None,
        'n_estim': n_estim,
        'box_type': box_type,
    }
    model3_info = {
        'model_name': model_name,
        'model_params': C_value,
        'rbfsampler_gamma': None,
        'nystroem_gamma': None,
        'sampler_name': 'nystroem',
        'pca_bool': False,
        'pca_first': None,
        'n_estim': n_estim,
        'box_type': box_type,
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
             description=f'Linear SVM ({dts_name})')
    d2 = exp(
        model_info=model2_info,
        tunning_params={},
        data_train=data_train,
        data_test=data_test,
        target_train=target_train,
        target_test=target_test,
        description=f'Linear SVM with RFF Black Bag ({dts_name})')
    d3 = exp(
        model_info=model3_info,
        tunning_params={},
        data_train=data_train,
        data_test=data_test,
        target_train=target_train,
        target_test=target_test,
        description=f'Linear SVM with Nystroem Black Bag ({dts_name})')

    store_exp(d1, d2, d3, exp_code=exp_code, dts_name=dts_name)


def exp3_1(dts_name):
    # Logit grey bag contra logit grey ensemnble
    # 2_3 contra 2_4
    exp_code = '3_1'
    prefix = 'experimental_results'
    dir1 = '2_3'
    dir2 = '2_4'
    path1 = f'{prefix}/{dir1}/{dts_name}.json'
    path2 = f'{prefix}/{dir2}/{dts_name}.json'

    with open(path1, 'r') as f:
        dic_list1 = json.load(f)
    with open(path2, 'r') as f:
        dic_list2 = json.load(f)

    label1_1 = "logit rff grey_bag 50 estims."
    label1_2 = "logit nystroem grey_bag 50 estims."
    label2_1 = "logit rff grey_ens 50 estims."
    label2_2 = "logit nystroem grey_ens 50 estims."

    first_labels = [label1_1, label1_2]
    second_labels = [label2_1, label2_2]

    dics1 = [i for i in dic_list1 if i['label'] in first_labels]
    dics2 = [i for i in dic_list2 if i['label'] in second_labels]

    store_exp(*dics1, *dics2, exp_code=exp_code, dts_name=dts_name)


def exp3_2(dts_name):
    exp_code = '3_2'
    model_name = 'logit'
    C_value = {'C': 1000}
    # box_type = 'black_bag'
    n_estim = 1
    # C_values = [10**i for i in range(4)]
    # tunning_params = {'C': C_values}
    tunning_params = {}
    model1_info = {
        'model_name': model_name,
        'model_params': C_value,
        'rbfsampler_gamma': None,
        'nystroem_gamma': None,
        'sampler_name': 'rbf',
        'pca_bool': False,
        'pca_first': None,
        'n_estim': n_estim,
        'box_type': 'black_bag',
    }
    model2_info = {
        'model_name': model_name,
        'model_params': C_value,
        'rbfsampler_gamma': None,
        'nystroem_gamma': None,
        'sampler_name': 'nystroem',
        'pca_bool': False,
        'pca_first': None,
        'n_estim': n_estim,
        'box_type': 'black_bag',
    }

    model3_info = {
        'model_name': model_name,
        'model_params': C_value,
        'rbfsampler_gamma': None,
        'nystroem_gamma': None,
        'sampler_name': 'rbf',
        'pca_bool': False,
        'pca_first': None,
        'n_estim': n_estim,
        'box_type': 'black_ens',
    }
    model4_info = {
        'model_name': model_name,
        'model_params': C_value,
        'rbfsampler_gamma': None,
        'nystroem_gamma': None,
        'sampler_name': 'nystroem',
        'pca_bool': False,
        'pca_first': None,
        'n_estim': n_estim,
        'box_type': 'black_ens',
    }

    data = get_data(dataset_name=dts_name, prop_train=2/3, n_ins=5000)
    data_train = data['data_train']
    data_test = data['data_test']
    target_train = data['target_train']
    target_test = data['target_test']

    d1 = exp(
        model_info=model1_info,
        tunning_params=tunning_params,
        data_train=data_train,
        data_test=data_test,
        target_train=target_train,
        target_test=target_test,
        description=f'Logit black_bag with RFF without regul. ({dts_name})')
    d2 = exp(
        model_info=model2_info,
        tunning_params=tunning_params,
        data_train=data_train,
        data_test=data_test,
        target_train=target_train,
        target_test=target_test,
        description=f'Logit black_bag with Nys without regul. ({dts_name})')

    d3 = exp(
        model_info=model3_info,
        tunning_params=tunning_params,
        data_train=data_train,
        data_test=data_test,
        target_train=target_train,
        target_test=target_test,
        description=f'Logit black_ens with RFF without regul. ({dts_name})')
    d4 = exp(
        model_info=model4_info,
        tunning_params=tunning_params,
        data_train=data_train,
        data_test=data_test,
        target_train=target_train,
        target_test=target_test,
        description=f'Logit black_ens with Nys without regul. ({dts_name})')

    store_exp(d1, d2, d3, d4, exp_code=exp_code, dts_name=dts_name)


def exp3_3(dts_name):
    # Linear SVM grey bag contra Linear SVM grey ensemnble
    # 2_3 contra 2_4
    exp_code = '3_3'
    prefix = 'experimental_results'
    dir1 = '2_7'
    dir2 = '2_8'
    path1 = f'{prefix}/{dir1}/{dts_name}.json'
    path2 = f'{prefix}/{dir2}/{dts_name}.json'

    with open(path1, 'r') as f:
        dic_list1 = json.load(f)
    with open(path2, 'r') as f:
        dic_list2 = json.load(f)

    label1_1 = "linear_svc rff grey_bag 50 estims."
    label1_2 = "linear_svc nystroem grey_bag 50 estims."
    label2_1 = "linear_svc rff grey_ens 50 estims."
    label2_2 = "linear_svc nystroem grey_ens 50 estims."

    first_labels = [label1_1, label1_2]
    second_labels = [label2_1, label2_2]

    dics1 = [i for i in dic_list1 if i['label'] in first_labels]
    dics2 = [i for i in dic_list2 if i['label'] in second_labels]

    store_exp(*dics1, *dics2, exp_code=exp_code, dts_name=dts_name)


def exp3_4(dts_name):
    exp_code = '3_4'
    model_name = 'linear_svc'
    # C_value = {'C': 1000}
    # box_type = 'black_bag'
    n_estim = 1
    C_values = [10**i for i in range(4)]
    tunning_params = {'C': C_values}
    # tunning_params = {}
    model1_info = {
        'model_name': model_name,
        'model_params': {},
        'rbfsampler_gamma': None,
        'nystroem_gamma': None,
        'sampler_name': 'rbf',
        'pca_bool': False,
        'pca_first': None,
        'n_estim': n_estim,
        'box_type': 'black_bag',
    }
    model2_info = {
        'model_name': model_name,
        'model_params': {},
        'rbfsampler_gamma': None,
        'nystroem_gamma': None,
        'sampler_name': 'nystroem',
        'pca_bool': False,
        'pca_first': None,
        'n_estim': n_estim,
        'box_type': 'black_bag',
    }

    model3_info = {
        'model_name': model_name,
        'model_params': {},
        'rbfsampler_gamma': None,
        'nystroem_gamma': None,
        'sampler_name': 'rbf',
        'pca_bool': False,
        'pca_first': None,
        'n_estim': n_estim,
        'box_type': 'black_ens',
    }
    model4_info = {
        'model_name': model_name,
        'model_params': {},
        'rbfsampler_gamma': None,
        'nystroem_gamma': None,
        'sampler_name': 'nystroem',
        'pca_bool': False,
        'pca_first': None,
        'n_estim': n_estim,
        'box_type': 'black_ens',
    }

    data = get_data(dataset_name=dts_name, prop_train=2/3, n_ins=5000)
    data_train = data['data_train']
    data_test = data['data_test']
    target_train = data['target_train']
    target_test = data['target_test']

    d1 = exp(
        model_info=model1_info,
        tunning_params=tunning_params,
        data_train=data_train,
        data_test=data_test,
        target_train=target_train,
        target_test=target_test,
        description=f'Linear-SVC black_bag with RFF without regul. ({dts_name})')
    d2 = exp(
        model_info=model2_info,
        tunning_params=tunning_params,
        data_train=data_train,
        data_test=data_test,
        target_train=target_train,
        target_test=target_test,
        description=f'Linear-SVC black_bag with Nys without regul. ({dts_name})')

    d3 = exp(
        model_info=model3_info,
        tunning_params=tunning_params,
        data_train=data_train,
        data_test=data_test,
        target_train=target_train,
        target_test=target_test,
        description=f'Linear-SVC black_ens with RFF without regul. ({dts_name})')
    d4 = exp(
        model_info=model4_info,
        tunning_params=tunning_params,
        data_train=data_train,
        data_test=data_test,
        target_train=target_train,
        target_test=target_test,
        description=f'Linear-SVC black_ens with Nys without regul. ({dts_name})')

    store_exp(d1, d2, d3, d4, exp_code=exp_code, dts_name=dts_name)


def exp4_1(dts_name):
    exp_code = '4_1'
    model_name = 'dt'
    box_type = 'none'
    # dt_params = {
    #     'splitter': 'best',
    #     'max_features': 'sqrt',
    # }
    n_estim = None
    min_id = [0, .1, .2, .5, 1]
    tunning_params = {'min_impurity_decrease': min_id}
    model1_info = {
        'model_name': model_name,
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
        'model_name': model_name,
        'model_params': {},
        'rbfsampler_gamma': None,
        'nystroem_gamma': None,
        'sampler_name': 'rbf',
        'pca_bool': False,
        'pca_first': None,
        'n_estim': n_estim,
        'box_type': box_type,
    }
    model3_info = {
        'model_name': model_name,
        'model_params': {},
        'rbfsampler_gamma': None,
        'nystroem_gamma': None,
        'sampler_name': 'nystroem',
        'pca_bool': False,
        'pca_first': None,
        'n_estim': n_estim,
        'box_type': box_type,
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
             description=f'DT ({dts_name})')
    d2 = exp(
        model_info=model2_info,
        tunning_params=tunning_params,
        data_train=data_train,
        data_test=data_test,
        target_train=target_train,
        target_test=target_test,
        description=f'DT with RFF ({dts_name})')
    d3 = exp(
        model_info=model3_info,
        tunning_params=tunning_params,
        data_train=data_train,
        data_test=data_test,
        target_train=target_train,
        target_test=target_test,
        description=f'DT with Nystroem ({dts_name})')

    store_exp(d1, d2, d3, exp_code=exp_code, dts_name=dts_name)


def exp4_2(dts_name):
    exp_code = '4_2'
    model_name = 'dt'
    box_type = 'black_bag'
    dt_params = {
        'splitter': 'best',
        'max_features': 'sqrt',
    }
    n_estim = 50
    min_id = [0, .1, .2, .5, 1]
    tunning_params = {'min_impurity_decrease': min_id}
    model1_info = {
        'model_name': model_name,
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
        'model_name': model_name,
        'model_params': dt_params,
        'rbfsampler_gamma': None,
        'nystroem_gamma': None,
        'sampler_name': 'rbf',
        'pca_bool': False,
        'pca_first': None,
        'n_estim': n_estim,
        'box_type': box_type,
    }
    model3_info = {
        'model_name': model_name,
        'model_params': dt_params,
        'rbfsampler_gamma': None,
        'nystroem_gamma': None,
        'sampler_name': 'nystroem',
        'pca_bool': False,
        'pca_first': None,
        'n_estim': n_estim,
        'box_type': box_type,
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
             description=f'DT ({dts_name})')
    d2 = exp(
        model_info=model2_info,
        tunning_params={},
        data_train=data_train,
        data_test=data_test,
        target_train=target_train,
        target_test=target_test,
        description=f'DT with RFF Black Bag ({dts_name})')
    d3 = exp(
        model_info=model3_info,
        tunning_params={},
        data_train=data_train,
        data_test=data_test,
        target_train=target_train,
        target_test=target_test,
        description=f'DT with Nystroem Black Bag ({dts_name})')

    store_exp(d1, d2, d3, exp_code=exp_code, dts_name=dts_name)


def exp4_3(dts_name):
    exp_code = '4_3'
    model_name = 'dt'
    box_type = 'black_ens'
    dt_params = {
        'splitter': 'best',
        'max_features': 'sqrt',
    }
    n_estim = 50
    min_id = [0, .1, .2, .5, 1]
    tunning_params = {'min_impurity_decrease': min_id}
    model1_info = {
        'model_name': model_name,
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
        'model_name': model_name,
        'model_params': dt_params,
        'rbfsampler_gamma': None,
        'nystroem_gamma': None,
        'sampler_name': 'rbf',
        'pca_bool': False,
        'pca_first': None,
        'n_estim': n_estim,
        'box_type': box_type,
    }
    model3_info = {
        'model_name': model_name,
        'model_params': dt_params,
        'rbfsampler_gamma': None,
        'nystroem_gamma': None,
        'sampler_name': 'nystroem',
        'pca_bool': False,
        'pca_first': None,
        'n_estim': n_estim,
        'box_type': box_type,
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
             description=f'DT ({dts_name})')
    d2 = exp(
        model_info=model2_info,
        tunning_params={},
        data_train=data_train,
        data_test=data_test,
        target_train=target_train,
        target_test=target_test,
        description=f'DT with RFF Black Bag ({dts_name})')
    d3 = exp(
        model_info=model3_info,
        tunning_params={},
        data_train=data_train,
        data_test=data_test,
        target_train=target_train,
        target_test=target_test,
        description=f'DT with Nystroem Black Bag ({dts_name})')

    store_exp(d1, d2, d3, exp_code=exp_code, dts_name=dts_name)


def exp4_4(dts_name):
    exp_code = '4_4'
    model_name = 'dt'
    box_type = 'grey_bag'
    dt_params = {
        'splitter': 'best',
        'max_features': 'sqrt',
    }
    n_estim = 50
    min_id = [0, .1, .2, .5, 1]
    tunning_params = {'min_impurity_decrease': min_id}
    model1_info = {
        'model_name': model_name,
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
        'model_name': model_name,
        'model_params': dt_params,
        'rbfsampler_gamma': None,
        'nystroem_gamma': None,
        'sampler_name': 'rbf',
        'pca_bool': False,
        'pca_first': None,
        'n_estim': n_estim,
        'box_type': box_type,
    }
    model3_info = {
        'model_name': model_name,
        'model_params': dt_params,
        'rbfsampler_gamma': None,
        'nystroem_gamma': None,
        'sampler_name': 'nystroem',
        'pca_bool': False,
        'pca_first': None,
        'n_estim': n_estim,
        'box_type': box_type,
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
             description=f'DT ({dts_name})')
    d2 = exp(
        model_info=model2_info,
        tunning_params={},
        data_train=data_train,
        data_test=data_test,
        target_train=target_train,
        target_test=target_test,
        description=f'DT with RFF Black Bag ({dts_name})')
    d3 = exp(
        model_info=model3_info,
        tunning_params={},
        data_train=data_train,
        data_test=data_test,
        target_train=target_train,
        target_test=target_test,
        description=f'DT with Nystroem Black Bag ({dts_name})')

    store_exp(d1, d2, d3, exp_code=exp_code, dts_name=dts_name)


def exp4_5(dts_name):
    exp_code = '4_5'
    model_name = 'dt'
    box_type = 'grey_ens'
    dt_params = {
        'splitter': 'best',
        'max_features': 'sqrt',
    }
    n_estim = 50
    min_id = [0, .1, .2, .5, 1]
    tunning_params = {'min_impurity_decrease': min_id}
    model1_info = {
        'model_name': model_name,
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
        'model_name': model_name,
        'model_params': dt_params,
        'rbfsampler_gamma': None,
        'nystroem_gamma': None,
        'sampler_name': 'rbf',
        'pca_bool': False,
        'pca_first': None,
        'n_estim': n_estim,
        'box_type': box_type,
    }
    model3_info = {
        'model_name': model_name,
        'model_params': dt_params,
        'rbfsampler_gamma': None,
        'nystroem_gamma': None,
        'sampler_name': 'nystroem',
        'pca_bool': False,
        'pca_first': None,
        'n_estim': n_estim,
        'box_type': box_type,
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
             description=f'DT ({dts_name})')
    d2 = exp(
        model_info=model2_info,
        tunning_params={},
        data_train=data_train,
        data_test=data_test,
        target_train=target_train,
        target_test=target_test,
        description=f'DT with RFF Black Bag ({dts_name})')
    d3 = exp(
        model_info=model3_info,
        tunning_params={},
        data_train=data_train,
        data_test=data_test,
        target_train=target_train,
        target_test=target_test,
        description=f'DT with Nystroem Black Bag ({dts_name})')

    store_exp(d1, d2, d3, exp_code=exp_code, dts_name=dts_name)
