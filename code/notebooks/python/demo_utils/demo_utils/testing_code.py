#!/usr/bin/python3
from demo_utils.general import SUPPORTED_DATASETS
# from IPython.display import Markdown as md
# from demo_utils.get_hyper_params import get_hyper_params
# from demo_utils.learning import get_model
# from sklearn.model_selection import GridSearchCV
# from demo_utils.general import get_data
# import time
# from demo_utils.general import gamest
# from demo_utils.general import get_label
import json
import matplotlib.pyplot as plt
import numpy as np
# from matplotlib import rcParams
from demo_utils.experiment_functions import exp1_1
from demo_utils.experiment_functions import exp2_1

from demo_utils.experiment_functions import exp2_2
from demo_utils.experiment_functions import exp2_3
from demo_utils.experiment_functions import exp2_4
from demo_utils.experiment_functions import exp2_5
from demo_utils.experiment_functions import exp2_6
from demo_utils.experiment_functions import exp2_7
from demo_utils.experiment_functions import exp2_8
from demo_utils.experiment_functions import exp3_1
from demo_utils.experiment_functions import exp3_2
from demo_utils.experiment_functions import exp3_3
from demo_utils.experiment_functions import exp3_4
from demo_utils.experiment_functions import exp4_1
from demo_utils.experiment_functions import exp4_2
from demo_utils.experiment_functions import exp4_3
from demo_utils.experiment_functions import exp4_4
from demo_utils.experiment_functions import exp4_5


# Aquí va el código que tiene que ejecutar los tests, mirando el tiempo,
# y todos los útils que tiene


def get_experiment(exp_code):
    '''
    Retorna una función que correrá el experimento indicado
    A la función habrá que pasarle nombre del dataset en el que
    se quiere ejecutar
    '''
    d = {
        '1_1': exp1_1,
        '2_1': exp2_1,

        '2_2': exp2_2,
        '2_3': exp2_3,
        '2_4': exp2_4,
        '2_5': exp2_5,
        '2_6': exp2_6,
        '2_7': exp2_7,
        '2_8': exp2_8,
        '3_1': exp3_1,
        '3_2': exp3_2,
        '3_3': exp3_3,
        '3_4': exp3_4,
        '4_1': exp4_1,
        '4_2': exp4_2,
        '4_3': exp4_3,
        '4_4': exp4_4,
        '4_5': exp4_5,
    }
    return d[exp_code]


def generate_png(exp_code, dts_name, labels, train_errors,
                 test_errors, times):
    '''
    Recibe la información necesaria (datos) y genera una imagen con
    una gráfica que lo representa
    '''
    mytimes = [-1 * i for i in times]
    filename = f'{exp_code}/{dts_name}'
    o_filename = f'experimental_graphs/{filename}.png'

    N = len(train_errors)
    ind = np.arange(N)
    width = 0.35

    fig, (x_errors, x_times) = plt.subplots(2, 1, sharex=True)
    fig.suptitle(dts_name)

    x_errors.bar(x=ind, height=train_errors, width=-width, align='edge',
                 label='Train', color='#ef5045')
    x_errors.bar(x=ind, height=test_errors, width=width,
                 align='edge', label='Test', color='#10cc35')

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

    x_errors.legend(loc=(.825, 1.02))

    plt.savefig(o_filename, bbox_inches="tight")


def generate_one_graph(exp_code, dts_name):
    '''
    Lee el fichero correspondiente y genera una imagen con la gráfica
    correspondiente
    '''
    filename = f'{exp_code}/{dts_name}'
    i_filename = f'experimental_results/{filename}.json'
    # o_filename = f'experimental_graphs/{filename}.png'

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


def generate_results(exp_code):
    foo = get_experiment(exp_code=exp_code)
    for dts_name in SUPPORTED_DATASETS:
        foo(dts_name=dts_name)


def generate_graphs(exp_code):
    '''
    Asume que ya están creados los json con la información, y genera
    las imágenes con las gráficas correspondientes
    '''
    for dts_name in SUPPORTED_DATASETS:
        generate_one_graph(exp_code=exp_code, dts_name=dts_name)


def whole_exp(exp_code):
    '''
    Genera los datos de los experimentos y genera también las imágenes con
    las gráficas
    '''
    generate_results(exp_code=exp_code)
    generate_graphs(exp_code=exp_code)


def graph_all_modes_all_datasets():
    for dts_name in SUPPORTED_DATASETS:
        graph_all_modes(dts_name=dts_name)


def graph_all_modes(dts_name):
    '''
    Asume que ya están realizados todos los experimentos y están guardados
    en su correspondiente json
    Genera una gráfica donde se muestran los errores y los tiempos de todos
    y cada uno de los modelos probados con el dts_name especificado

    Los modelos que tiene que mostrar son:

    RBF-SVC
    DT
    Logit
    Linear-SVC

    DT RFF y Nys
    Logit RFF y Nys
    Linear-SVC RFF y Nys

    DT Black Bag
    DT Grey Bag
    DT Black Ens
    DT Grey Ens

    Logit Black Bag
    Logit Grey Bag
    Logit Grey Ens

    Linear-SVC Black Bag
    Linear-SVC Grey Bag
    Linear-SVC Grey Ens
    '''
    locs = [
        ("1_1", "rbf_svc "),
        ("4_1", "dt "),
        ("2_1", "logit "),
        ("2_5", "linear_svc "),

        ("4_1", "dt rff "),
        ("4_1", "dt nystroem "),
        ("2_1", "logit rff "),
        ("2_1", "logit nystroem "),
        ("2_5", "linear_svc rff "),
        ("2_5", "linear_svc nystroem "),
        ("4_2", "dt rff black_bag 50 estims."),
        ("4_2", "dt nystroem black_bag 50 estims."),

        ("4_4", "dt rff grey_bag 50 estims."),
        ("4_4", "dt nystroem grey_bag 50 estims."),

        ("4_3", "dt rff black_ens 50 estims."),
        ("4_3", "dt nystroem black_ens 50 estims."),

        ("4_5", "dt rff grey_ens 50 estims."),
        ("4_5", "dt nystroem grey_ens 50 estims."),

        ("2_2", "logit rff black_bag 50 estims."),
        ("2_2", "logit nystroem black_bag 50 estims."),

        ("2_3", "logit rff grey_bag 50 estims."),
        ("2_3", "logit nystroem grey_bag 50 estims."),

        ("2_4", "logit rff grey_ens 50 estims."),
        ("2_4", "logit nystroem grey_ens 50 estims."),

        ("2_6", "linear_svc rff black_bag 50 estims."),
        ("2_6", "linear_svc nystroem black_bag 50 estims."),

        ("2_7", "linear_svc rff grey_bag 50 estims."),
        ("2_7", "linear_svc nystroem grey_bag 50 estims."),

        ("2_8", "linear_svc rff grey_ens 50 estims."),
        ("2_8", "linear_svc nystroem grey_ens 50 estims."),
    ]

    labels = []
    train_errors = []
    test_errors = []
    times = []

    for i in locs:
        exp = i[0]
        lab = i[1]

        i_filename = f'experimental_results/{exp}/{dts_name}.json'
        with open(i_filename, 'r') as f:
            dic_list = json.load(f)
        dic = [d for d in dic_list if d['label'] == lab][0]
        labels.append(dic['label'])
        train_errors.append(1 - dic['train_score'])
        test_errors.append(1 - dic['test_score'])
        times.append(dic['time'])

    indices = np.argsort(test_errors)

    labels2 = [labels[i] for i in indices]
    train_errors2 = [train_errors[i] for i in indices]
    test_errors2 = [test_errors[i] for i in indices]
    times2 = [times[i] for i in indices]

    # my_new_generate_png(dts_name=dts_name,
    #                     labels=labels,
    #                     train_errors=train_errors,
    #                     test_errors=test_errors,
    #                     times=times)

    my_new_generate_png(dts_name=dts_name,
                        labels=labels2,
                        train_errors=train_errors2,
                        test_errors=test_errors2,
                        times=times2)


def generate_all_fashion():
    prefix = 'experimental_results/fashion'
    files_names = [
        'dt.json',
        'dt_rff_grey_bag.json',
        'dt_rff.json',
        'linear_svc.json',
        'linear_svc_rff.json',
        'logit.json',
        'logit_rff_grey_bag.json',
        'logit_rff.json',
        'rbf_svc.json',
    ]
    labels = []
    train_errors = []
    test_errors = []
    times = []
    for file in files_names:
        path = f'{prefix}/{file}'
        with open(path, 'r') as f:
            dic_list = json.load(f)
            dic = dic_list[0]
        labels.append(dic['label'])
        train_errors.append(1 - dic['train_score'])
        test_errors.append(1 - dic['test_score'])
        times.append(dic['time'])

        indices = np.argsort(test_errors)

        labels2 = [labels[i] for i in indices]
        train_errors2 = [train_errors[i] for i in indices]
        test_errors2 = [test_errors[i] for i in indices]
        times2 = [times[i] for i in indices]

        # Ojo, que no lo guarda en el sitio que nos gustaría

        my_new_generate_png(dts_name='fashion',
                            labels=labels2,
                            train_errors=train_errors2,
                            test_errors=test_errors2,
                            times=times2)



def my_new_generate_png(dts_name, labels, train_errors, test_errors, times):
    '''
    Recibe la información necesaria (datos) y genera una imagen con
    una gráfica que lo representa
    '''
    mytimes = [-1 * i for i in times]
    # filename = f'{exp_code}/{dts_name}'
    # o_filename = f'experimental_graphs/{filename}.png'
    o_filename = f'experimental_graphs/all_results/{dts_name}.png'

    N = len(train_errors)
    ind = np.arange(N)
    width = 0.80
    ind = ind * 2

    fig, (x_errors, x_times) = plt.subplots(2, 1, sharex=True, figsize=(18, 5))
    fig.suptitle(dts_name)

    x_errors.bar(x=ind, height=train_errors, width=-width, align='edge',
                 label='Train', color='#ef5045')
    x_errors.bar(x=ind, height=test_errors, width=width,
                 align='edge', label='Test', color='#10cc35')

    x_errors.set_ylabel('Error')

    # x_errors.set_xticks(ind)
    # x_errors.set_xticklabels(labels)

    x_times.bar(x=ind, height=mytimes, width=2*width, color='#49dfed')
    x_times.set_ylabel('Time (s)')
    plt.xticks(ind, labels)

    ticks = x_times.get_yticks()
    x_times.set_yticklabels([int(abs(tick)) for tick in ticks])

    degree = 90
    x_errors.tick_params(rotation=degree)
    x_times.tick_params(rotation=degree)

    x_times.tick_params(axis='y', reset=True)
    x_errors.tick_params(axis='y', reset=True)

    x_errors.grid(True)
    x_times.grid(True)

    x_errors.legend(loc=(.825, 1.02))

    plt.savefig(o_filename, bbox_inches="tight")
