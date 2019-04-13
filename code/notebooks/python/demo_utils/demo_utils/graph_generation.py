#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import json
import os
from demo_utils.general import SUPPORTED_DATASETS

# Contiene las funciones que asumen que los cálculos están
# hechos y que generan los las gráficas adecuadas


def beaut_label(label):
    changes = [('rbf', 'RBF'),
               ('svc', 'SVM'),
               ('dt', 'DT'),
               ('logit', 'LogReg.'),
               ('rff', 'RFF'),
               ('nystroem', 'Nys.'),
               ('linear_', ''),
               ('black', 'Black'),
               ('grey', 'White'),
               ('_bag', 'Bag'),
               ('_ens', 'Ens'),
               ('estims', 'est'), ]
    for o, n in changes:
        label = label.replace(o, n)
    return label


def generate_png(filename, dts_name, labels, train_errors,
                 test_errors, times):
    '''
    Recibe la información necesaria (datos) y genera una imagen con
    una gráfica que lo representa

    Parameters
    ==========
    filename : str
        El nombre del fichero (debería terminar en .png) en el que guardar
        la imagen. Asume un current directory de demo_utils/demo_utils/
    '''
    mytimes = [-1 * i for i in times]
    # filename = f'{exp_code}/{dts_name}'
    # o_filename = f'experimental_graphs/{filename}.png'

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

    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    plt.savefig(filename, bbox_inches="tight")
    plt.close()


def tmp(filenames, labels, out_filename, dts_name):
    '''

    Le dan una lista de documentos y labels, y genera una imagen
    con que tocan
    filenames y labes son listas de strings, con el mismo tamaño
    '''

    train_errors = []
    test_errors = []
    times = []
    for i, _ in enumerate(filenames):
        with open(filenames[i], 'r') as f:
            print(f'Reading {filenames[i]}...')
            dic_list = json.load(f)
        dic = [d for d in dic_list if d['label'] == labels[i]][0]
        train_errors.append(1 - dic['train_score'])
        test_errors.append(1 - dic['test_score'])
        times.append(dic['time'])

    b_labels = [beaut_label(i) for i in labels]

    generate_png(filename=out_filename,
                 dts_name=dts_name,
                 labels=b_labels,
                 train_errors=train_errors,
                 test_errors=test_errors,
                 times=times)

# def take_one_bar(filename, lab):
#     '''
#     La das el nombre de un json para consultar y el label que tiene
#     que coger, y retorna
#     (train_error, test_error, time)
#     '''
#     with open(filename, 'r') as f:
#         dic_list = json.load(f)
#     dic = [d for d in dic_list if d['label'] == lab][0]
#     train_score = dic['train_score']
#     test_score = dic['test_score']
#     time = dic['time']
#
#     train_error = 1 - train_score
#     pass


ficheros = {
    # '1.1': [(fichero, label), (fichero, label), (fichero, label)]
    # (exp, filename): [(fichero, label), (fichero, label), (fichero, label)],

    ('1_1', '.'): [('1_1/{}.json', "rbf_svc "),
                   ('2_5/{}.json', "linear_svc "),
                   ('1_1/{}.json', "linear_svc rff "),
                   ('1_1/{}.json', "linear_svc nystroem "), ],

    ('2_1', '.'): [('2_1/{}.json', "logit "),
                   ('2_1/{}.json', "logit rff "),
                   ('2_1/{}.json', "logit nystroem "), ],
    ('2_2', 'rff'): [('2_1/{}.json', "logit "),
                     ('2_2/{}.json', "logit rff black_bag 50 estims."),
                     ('2_3/{}.json', "logit rff grey_bag 50 estims."),
                     ('2_4/{}.json', "logit rff grey_ens 50 estims."), ],
    ('2_2', 'nys'): [('2_1/{}.json', "logit "),
                     ('2_2/{}.json', "logit nystroem black_bag 50 estims."),
                     ('2_3/{}.json', "logit nystroem grey_bag 50 estims."),
                     ('2_4/{}.json', "logit nystroem grey_ens 50 estims."), ],
    ('2_2', 'aux'): [('2_1/{}.json', "logit rff "),
                     ('2_3/{}.json', "logit rff grey_bag 50 estims."),
                     ('2_1/{}.json', "logit nystroem "),
                     ('2_2/{}.json', "logit nystroem black_bag 50 estims."), ],
    ('2_3', '.'): [('2_5/{}.json', "linear_svc "),
                   ('1_1/{}.json', "linear_svc rff "),
                   ('1_1/{}.json', "linear_svc nystroem "), ],
    ('2_4', 'rff'): [('2_5/{}.json', "linear_svc "),
                     ('2_6/{}.json', "linear_svc rff black_bag 50 estims."),
                     ('2_7/{}.json', "linear_svc rff grey_bag 50 estims."),
                     ('2_8/{}.json', "linear_svc rff grey_ens 50 estims."), ],
    ('2_4', 'nys'): [('2_5/{}.json', "linear_svc "),
                     ('2_6/{}.json', "linear_svc nystroem black_bag 50 estims."),
                     ('2_7/{}.json', "linear_svc nystroem grey_bag 50 estims."),
                     ('2_8/{}.json', "linear_svc nystroem grey_ens 50 estims."), ],
    ('2_4', 'aux'): [('1_1/{}.json', "linear_svc rff "),
                     ('2_6/{}.json', "linear_svc rff black_bag 50 estims."),
                     ('1_1/{}.json', "linear_svc nystroem "),
                     ('2_6/{}.json', "linear_svc nystroem black_bag 50 estims."), ],

    ('3_1', '.'): [('2_3/{}.json', "logit rff grey_bag 50 estims."),
                   ('2_4/{}.json', "logit rff grey_ens 50 estims."),
                   ('2_3/{}.json', "logit nystroem grey_bag 50 estims."),
                   ('2_8/{}.json', "linear_svc rff grey_ens 50 estims."),],
    ('3_2', '.'): [('3_2/{}.json', "logit rff black_bag 1 estims."),
                   ('3_2/{}.json', "logit rff black_ens 1 estims."),
                   ('3_2/{}.json', "logit nystroem black_bag 1 estims."),
                   ('3_2/{}.json', "logit nystroem black_ens 1 estims."), ],
    ('3_3', '.'): [('2_7/{}.json', "linear_svc rff grey_bag 50 estims."),
                   ('2_8/{}.json', "linear_svc rff grey_ens 50 estims."),
                   ('2_7/{}.json', "linear_svc nystroem grey_bag 50 estims."),
                   ('2_8/{}.json', "linear_svc nystroem grey_ens 50 estims."), ],
    ('3_4', '.'): [('3_4/{}.json', "linear_svc rff black_bag 1 estims."),
                   ('3_4/{}.json', "linear_svc rff black_ens 1 estims."),
                   ('3_4/{}.json', "linear_svc nystroem black_bag 1 estims."),
                   ('3_4/{}.json', "linear_svc nystroem black_ens 1 estims."), ],

    ('4_1', '.'): [('4_1/{}.json', "dt "),
                   ('4_1/{}.json', "dt rff "),
                   ('4_1/{}.json', "dt nystroem "), ],
    ('4_2', 'rff'): [('random_forest/{}.json', "RF "),
                     ('4_2/{}.json', "dt rff black_bag 50 estims."),
                     ('4_3/{}.json', "dt rff black_ens 50 estims."),
                     ('4_4/{}.json', "dt rff grey_bag 50 estims."),
                     ('4_5/{}.json', "dt rff grey_ens 50 estims."), ],
    ('4_2', 'nys'): [('random_forest/{}.json', "RF "),
                     ('4_2/{}.json', "dt nystroem black_bag 50 estims."),
                     ('4_3/{}.json', "dt nystroem black_ens 50 estims."),
                     ('4_4/{}.json', "dt nystroem grey_bag 50 estims."),
                     ('4_5/{}.json', "dt nystroem grey_ens 50 estims."), ],

}

def paint_all_results():
    '''
    Usando el diccionario ficheros, consulta todos los resultados y guarda
    las imágenes en un prefix

    El prefix asume que está trabajando sobre demo_utils/demo_utils
    '''
    PREFIX = 'new_experimental_graphs'
    for exp, dir in ficheros:
        new_dts_list = ['fashion_mnist'] + SUPPORTED_DATASETS
        # for dts_name in SUPPORTED_DATASETS:
        for dts_name in new_dts_list:
            out_filename = f'{PREFIX}/{exp}/{dir}/{dts_name}.png'
            names_labels = [('experimental_results/'+d.format(dts_name), l) for d, l in ficheros[(exp, dir)]]

            filenames, filelabels = zip(*names_labels)

            tmp(filenames=filenames,
                labels=filelabels,
                out_filename=out_filename,
                dts_name=dts_name)
