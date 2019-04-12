#!/usr/bin/python3

# Tiene ejecuciones para mnist con todas las instancias,
# y las guarda en su propio directorio


from mnist import MNIST

from demo_utils.general_experiment import exp
import numpy as np
# from demo_utils.general import gamest
from demo_utils.general_experiment import store_exp
from sklearn.preprocessing import StandardScaler


# def get_fashion_data():
def get_normal_mnist_data():
    data_dir = '/home/hobber/mnist'
    mndata = MNIST(data_dir)
    data_train, target_train = mndata.load_training()
    data_test, target_test = mndata.load_testing()

    scaler = StandardScaler()
    scaler.fit(data_train)
    data_train = scaler.transform(data_train)
    data_test = scaler.transform(data_test)

    d = {
        'data_train': np.array(data_train),
        'data_test': np.array(data_test),
        'target_train': np.array(target_train),
        'target_test': np.array(target_test),
    }
    return d


def temp_func():
    names = ['logit_rff_grey_bag',
             'dt_rff_grey_bag',
             'linear_svc_rff_grey_bag']
    for n in names:
        # general_fashion(n)
        general_normal_mnist(n)


# def general_fashion(model_name):
def general_normal_mnist(model_name):
    funcs = {
        'linear_svc': normal_mnist_linear_svc,
        'rbf_svc': normal_mnist_rbf_svc,
        'logit': normal_mnist_logit,
        'dt': normal_mnist_dt,
        'linear_svc_rff': normal_mnist_linear_svc_rff,
        'logit_rff': normal_mnist_logit_rff,
        'dt_rff': normal_mnist_dt_rff,

        'logit_rff_grey_bag': normal_mnist_logit_rff_grey_bag,
        'dt_rff_grey_bag': normal_mnist_dt_rff_grey_bag,
        'linear_svc_rff_grey_bag': normal_mnist_linear_svc_rff_grey_bag,
    }
    data = get_normal_mnist_data()
    data_train = data['data_train']
    data_test = data['data_test']
    target_train = data['target_train']
    target_test = data['target_test']

    call_func = funcs[model_name]

    call_func(data_train=data_train,
              data_test=data_test,
              target_train=target_train,
              target_test=target_test)


def normal_mnist_linear_svc_rff(data_train, data_test, target_train, target_test):
    # C_value = 50
    C_values = [0.5, 1, 5, 20, 50]
    tunning_params = {'C': C_values}
    # model_params = {'C': C_value}
    # tunning_params = {}
    model_params = {}

    model_info = {
        'model_name': 'linear_svc',
        'model_params': model_params,
        'rbfsampler_gamma': None,
        'nystroem_gamma': None,
        'sampler_name': 'rbf',
        'pca_bool': False,
        'pca_first': None,
        'n_estim': None,
        'box_type': 'none',
    }
    print('Empieza el experimento')
    d = exp(model_info=model_info,
            tunning_params=tunning_params,
            data_train=data_train,
            data_test=data_test,
            target_train=target_train,
            target_test=target_test,
            description='A linear SVM with RFF with normal mnist')
    print('Termina el experimento')

    store_exp(d, exp_code='normal_mnist', dts_name='linear_svc_rff')


def normal_mnist_linear_svc(data_train, data_test, target_train, target_test):
    # C_value = 50
    C_values = [0.5, 1, 5, 20, 50]
    tunning_params = {'C': C_values}
    # model_params = {'C': C_value}
    # tunning_params = {}
    model_params = {}

    model_info = {
        'model_name': 'linear_svc',
        'model_params': model_params,
        'rbfsampler_gamma': None,
        'nystroem_gamma': None,
        'sampler_name': 'identity',
        'pca_bool': False,
        'pca_first': None,
        'n_estim': None,
        'box_type': 'none',
    }
    print('Empieza el experimento')
    d = exp(model_info=model_info,
            tunning_params=tunning_params,
            data_train=data_train,
            data_test=data_test,
            target_train=target_train,
            target_test=target_test,
            description='A linear SVM with normal mnist')
    print('Termina el experimento')

    store_exp(d, exp_code='normal_mnist', dts_name='linear_svc')


def normal_mnist_logit_rff(data_train, data_test, target_train, target_test):
    C_value = 1000
    # C_values = [0.5, 1, 5, 20, 50]
    # tunning_params = {'C': C_values}
    model_params = {'C': C_value}
    tunning_params = {}

    model_info = {
        'model_name': 'logit',
        'model_params': model_params,
        'rbfsampler_gamma': None,
        'nystroem_gamma': None,
        'sampler_name': 'rbf',
        'pca_bool': False,
        'pca_first': None,
        'n_estim': None,
        'box_type': 'none',
    }
    print('Empieza el experimento')
    d = exp(model_info=model_info,
            tunning_params=tunning_params,
            data_train=data_train,
            data_test=data_test,
            target_train=target_train,
            target_test=target_test,
            description='A Logit with RFF with normal mnist')
    print('Termina el experimento')

    store_exp(d, exp_code='normal_mnist', dts_name='logit_rff')


def normal_mnist_logit(data_train, data_test, target_train, target_test):
    C_value = 1000
    # C_values = [0.5, 1, 5, 20, 50]
    # tunning_params = {'C': C_values}
    model_params = {'C': C_value}
    tunning_params = {}

    model_info = {
        'model_name': 'logit',
        'model_params': model_params,
        'rbfsampler_gamma': None,
        'nystroem_gamma': None,
        'sampler_name': 'identity',
        'pca_bool': False,
        'pca_first': None,
        'n_estim': None,
        'box_type': 'none',
    }
    print('Empieza el experimento')
    d = exp(model_info=model_info,
            tunning_params=tunning_params,
            data_train=data_train,
            data_test=data_test,
            target_train=target_train,
            target_test=target_test,
            description='A Logit with normal mnist')
    print('Termina el experimento')

    store_exp(d, exp_code='normal_mnist', dts_name='logit')


def normal_mnist_dt_rff(data_train, data_test, target_train, target_test):
    # min_id = 0.2
    min_id_values = [0, 0.1, 0.2, 0.5]
    tunning_params = {'min_impurity_decrease': min_id_values}
    # model_params = {'min_impurity_decrease': min_id}
    model_params = {}
    # tunning_params = {}

    model_info = {
        'model_name': 'dt',
        'model_params': model_params,
        'rbfsampler_gamma': None,
        'nystroem_gamma': None,
        'sampler_name': 'rbf',
        'pca_bool': False,
        'pca_first': None,
        'n_estim': None,
        'box_type': 'none',
    }
    print('Empieza el experimento')
    d = exp(model_info=model_info,
            tunning_params=tunning_params,
            data_train=data_train,
            data_test=data_test,
            target_train=target_train,
            target_test=target_test,
            description='A DT with RFF with normal mnist')
    print('Termina el experimento')

    store_exp(d, exp_code='normal_mnist', dts_name='dt_rff')


def normal_mnist_dt(data_train, data_test, target_train, target_test):
    # min_id = 0.2
    min_id_values = [0, 0.1, 0.2, 0.5]
    tunning_params = {'min_impurity_decrease': min_id_values}
    # model_params = {'min_impurity_decrease': min_id}
    model_params = {}
    # tunning_params = {}

    model_info = {
        'model_name': 'dt',
        'model_params': model_params,
        'rbfsampler_gamma': None,
        'nystroem_gamma': None,
        'sampler_name': 'identity',
        'pca_bool': False,
        'pca_first': None,
        'n_estim': None,
        'box_type': 'none',
    }
    print('Empieza el experimento')
    d = exp(model_info=model_info,
            tunning_params=tunning_params,
            data_train=data_train,
            data_test=data_test,
            target_train=target_train,
            target_test=target_test,
            description='A DT with normal mnist')
    print('Termina el experimento')

    store_exp(d, exp_code='normal_mnist', dts_name='dt')


def normal_mnist_rbf_svc(data_train, data_test, target_train, target_test):
    C_value = 20
    # C_values = [0.5, 1, 5, 20, 50]
    # tunning_params = {'C': C_values}
    model_params = {'C': C_value}
    tunning_params = {}

    model_info = {
        'model_name': 'rbf_svc',
        'model_params': model_params,
        'rbfsampler_gamma': None,
        'nystroem_gamma': None,
        'sampler_name': 'identity',
        'pca_bool': False,
        'pca_first': None,
        'n_estim': None,
        'box_type': 'none',
    }
    print('Empieza el experimento')
    d = exp(model_info=model_info,
            tunning_params=tunning_params,
            data_train=data_train,
            data_test=data_test,
            target_train=target_train,
            target_test=target_test,
            description='An RBF-SVM with normal mnist')
    print('Termina el experimento')

    store_exp(d, exp_code='normal_mnist', dts_name='rbf_svc')

######################
# Ensembles
######################


def normal_mnist_logit_rff_grey_bag(data_train, data_test, target_train, target_test):
    C_value = 1000
    # C_values = [0.5, 1, 5, 20, 50]
    # tunning_params = {'C': C_values}
    model_params = {'C': C_value}
    tunning_params = {}

    model_info = {
        'model_name': 'logit',
        'model_params': model_params,
        'rbfsampler_gamma': None,
        'nystroem_gamma': None,
        'sampler_name': 'rbf',
        'pca_bool': False,
        'pca_first': None,
        'n_estim': 50,
        'box_type': 'grey_bag',
    }
    print('Empieza el experimento')
    d = exp(model_info=model_info,
            tunning_params=tunning_params,
            data_train=data_train,
            data_test=data_test,
            target_train=target_train,
            target_test=target_test,
            description='A Logit with RFF Grey Bag with normal mnist')
    print('Termina el experimento')

    store_exp(d, exp_code='normal_mnist', dts_name='logit_rff_grey_bag')


def normal_mnist_linear_svc_rff_grey_bag(data_train, data_test, target_train, target_test):
    C_value = 100
    # C_values = [0.5, 1, 5, 20, 50]
    # tunning_params = {'C': C_values}
    model_params = {'C': C_value}
    tunning_params = {}
    # model_params = {}

    model_info = {
        'model_name': 'linear_svc',
        'model_params': model_params,
        'rbfsampler_gamma': None,
        'nystroem_gamma': None,
        'sampler_name': 'rbf',
        'pca_bool': False,
        'pca_first': None,
        'n_estim': 50,
        'box_type': 'grey_bag',
    }
    print('Empieza el experimento')
    d = exp(model_info=model_info,
            tunning_params=tunning_params,
            data_train=data_train,
            data_test=data_test,
            target_train=target_train,
            target_test=target_test,
            description='A linear SVM with RFF Grey Bag with normal mnist')
    print('Termina el experimento')

    store_exp(d, exp_code='normal_mnist', dts_name='linear_svc_rff_grey_bag')


def normal_mnist_dt_rff_grey_bag(data_train, data_test, target_train, target_test):
    # min_id = 0.2
    # min_id_values = [0, 0.1, 0.2, 0.5]
    # tunning_params = {'min_impurity_decrease': min_id_values}
    # model_params = {'min_impurity_decrease': min_id}
    model_params = {}
    tunning_params = {}

    model_info = {
        'model_name': 'dt',
        'model_params': model_params,
        'rbfsampler_gamma': None,
        'nystroem_gamma': None,
        'sampler_name': 'rbf',
        'pca_bool': False,
        'pca_first': None,
        'n_estim': 50,
        'box_type': 'grey_bag',
    }
    print('Empieza el experimento')
    d = exp(model_info=model_info,
            tunning_params=tunning_params,
            data_train=data_train,
            data_test=data_test,
            target_train=target_train,
            target_test=target_test,
            description='A DT with RFF Grey Bag with normal mnist')
    print('Termina el experimento')

    store_exp(d, exp_code='normal_mnist', dts_name='dt_rff_grey_bag')
