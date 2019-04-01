# TODO desde fuera de las demos, algo que me permita definir las
# non_interactive más fácilmente, con un lenguage simbólico

# TODO quizá me da problemas el 'None' o 'none'

# boxes = ['black_bag', 'grey_bag', 'black_ens', 'grey_ens', 'None']
# boxes = ['black_bag', 'grey_bag', 'black_ens', 'grey_ens', 'none']

import time

def simple_time_and_accuracy_test(dts_name,
                                  model_name,
                                  box_type,
                                  ):
    '''
    La función de abajo tiene que hacer 3 ejecuciones de esta, que solo hace una
    '''

def time_and_accuracy_test(dts_name,
                           ):
    '''
    Hace una ejecución del test que se indica en los parámetros y con
    algún tipo de estructura retorna el accuracy obtenido con cada uno de ellos
    y los tiempos de cross-validation, entrenamiento y respuesta
    '''


def auto_demo(model, dts):
    boxes = ['black_bag', 'grey_bag', 'black_ens', 'grey_ens', 'none']
    n_estim = 50
    range = (30, 600)
    dts_size = 2000
    m_base = {
        'model_name': model,
        'sampler_name': 'identity',
        'box_type': 'none',
        'n_estim': None,
        'pca': False,
        'pca_first': False
    }
    m = {
        'model_name': model,
        'sampler_name': 'rbf',
        'box_type': 'INVALID',
        'n_estim': n_estim,
        'pca': 'INVALID',
        'pca_first': 'INVALID',
    }

    # no pca
    m_1_1 = dict(m)
    # pca primero
    m_1_2 = dict(m)
    # pca después
    m_1_3 = dict(m)

    m_1_1['pca'] = False
    m_1_1['pca_first'] = True

    m_1_2['pca'] = True
    m_1_2['pca_first'] = True

    m_1_2['pca'] = True
    m_1_2['pca_first'] = False

    lista_mods = [m_1_1, m_1_2, m_1_3]

    lista_final_mods = []
    for bt in boxes:
        for model in lista_mods:
            gg = dict(model)
            gg['box_type'] = bt
            lista_final_mods.append(gg)
    lista_final_mods.append(m_base)

    info = {
        'models': lista_final_mods,
        'features_range': range,
        'dts_size': dts_size,
        'dts_name': dts
    }
    return info


###########################3


def auto_demo_fragmented(model, dts):
    boxes = ['black_bag', 'grey_bag', 'black_ens', 'grey_ens', 'none']
    n_estim = 10
    range = (30, 400)
    dts_size = 1000
    m_base = {
        'model_name': model,
        'sampler_name': 'identity',
        'box_type': 'none',
        'n_estim': None,
        'pca': False,
        'pca_first': False
    }
    m = {
        'model_name': model,
        'sampler_name': 'rbf',
        'box_type': 'INVALID',
        'n_estim': n_estim,
        'pca': 'INVALID',
        'pca_first': 'INVALID',
    }

    # no pca
    m_1_1 = dict(m)
    # pca primero
    m_1_2 = dict(m)
    # pca después
    m_1_3 = dict(m)

    m_1_1['pca'] = False
    m_1_1['pca_first'] = True

    m_1_2['pca'] = True
    m_1_2['pca_first'] = True

    m_1_2['pca'] = True
    m_1_2['pca_first'] = False

    lista_mods = [m_1_1, m_1_2, m_1_3]

    # lista_final_mods = []
    # for bt in boxes:
    #     minilist = []
    #     for model in lista_mods:
    #         gg = dict(model)
    #         gg['box_type'] = bt
    #         minilist.append(gg)
    #     lista_final_mods.append(minilist)
    # lista_final_mods.append(m_base)

    lista_final_mods = []
    for model in lista_mods:
        minilist = []
        for bt in boxes:
            gg = dict(model)
            gg['box_type'] = bt
            minilist.append(gg)
        lista_final_mods.append(minilist)

    info0 = {
        'models': [m_base] + lista_final_mods[0],
        'features_range': range,
        'dts_size': dts_size,
        'dts_name': dts
    }
    info1 = {
        'models': [m_base] + lista_final_mods[1],
        'features_range': range,
        'dts_size': dts_size,
        'dts_name': dts
    }
    info2 = {
        'models': [m_base] + lista_final_mods[2],
        'features_range': range,
        'dts_size': dts_size,
        'dts_name': dts
    }

    ret_list = [info0, info1, info2]
    return ret_list
    # return info
