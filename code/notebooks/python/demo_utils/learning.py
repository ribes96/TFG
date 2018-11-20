from sklearn.preprocessing import FunctionTransformer
from sklearn.kernel_approximation import Nystroem
from sklearn.kernel_approximation import RBFSampler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.ensemble import BaggingClassifier
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import numpy as np

from demo_utils.general import get_label
from demo_utils.general import get_data


# import pandas as pd


# currentwork
def get_model(model,
              sampler = None,
              pca = False,
              ensemble = None,
              box_type = None):
    '''
    Parameters
    ----------
    model: string, 'dt', 'linear_svc' or 'logit'
    sampler: string, 'rbf' or 'nystroem', or  None
    pca: bool
    ensemble: integer or None
    box_type: string, 'black', 'grey' or None (ignored if ensemble = None)

    Returns
    -------
    clf: a model with the parameters specified
    '''
    if model not in ['dt', 'linear_svc', 'logit']:
        raise ValueError('model {0} is not supported'.format(model))
    if sampler not in ['rbf', 'nystroem', None]:
        raise ValueError('sampler {0} is not supported'.format(sampler))
    if type(pca) != bool:
        raise ValueError('pca is a boolean')
    if type(ensemble) not in [int, type(None)]:
    # if not isinstance(ensemble, int) and ensemble != 'None':
        raise ValueError('Wrong value for ensemble')
    if isinstance(ensemble,int) and ensemble < 1:
        raise ValueError('Number of estimators must be greater than 0')
    if box_type not in ['black', 'grey', None]:
        raise ValueError("box_type must be 'black', 'grey' or None")
    if box_type is not None and ensemble is None:
        raise ValueError("box_type doesn't match with ensemble")

    #s = RBFSampler() if sampler == 'rbf' else
    #   Nystroem() if sampler == 'nystroem' else FunctionTransformer(None, validate = False)
    #p = PCA() if pca else FunctionTransformer(None, validate = False)
    #m = DecisionTreeClassifier() if model == 'dt'
    #      else LinearSVC() if model == 'linear_svc' else LogisticRegression()


    if sampler == 'rbf':
        s = RBFSampler(gamma = 0.2)
    elif sampler == 'nystroem':
        s = Nystroem(gamma = 0.2)
    elif sampler is None:
        s = FunctionTransformer(None, validate = False)
    else:
        # todo esto son pruebas
        print("Sampler no tiene un valor adeacuado")
        print("sampler: {}".format(sampler))
        raise ValueError("patata")



    if pca:
        p = PCA(n_components = 0.9, svd_solver = "full")
    else:
        p = FunctionTransformer(None, validate = False)

    if model == 'dt':
        m = DecisionTreeClassifier()
    elif model == 'linear_svc':
        m = LinearSVC(C = 1)
    elif model == 'logit':
        m = LogisticRegression(C = 1, multi_class = 'multinomial', solver = 'lbfgs')


    # if not ensemble:
    if ensemble is None:
        clf = Pipeline([
            ('sampler', s),
            ('pca', p),
            ('model', m),
        ])
    elif box_type == 'black':
        bag = BaggingClassifier(base_estimator = m, n_estimators = ensemble)
        clf = Pipeline([
            ('sampler', s),
            ('pca', p),
            ('model', bag),
        ])
    elif box_type == 'grey':
        pipe = Pipeline([
            ('sampler', s),
            ('pca', p),
            ('model', m),
        ])
        clf = BaggingClassifier(base_estimator = pipe, n_estimators = ensemble)

    return clf


# def get_label(model, sampler, pca, box_type, train_test):
#     '''Returns a string with the correct label
#     Parameters
#     ----------
#     model: string, model name
#     sampler: string, 'rbf' or 'nystroem', or  None
#     pca: bool
#     box_type: string, 'black', 'grey' or None
#     train_test: string, 'train' or 'test'
#     '''
#     if sampler is not None and sampler not in ['rbf', 'nystroem']:
#         raise ValueError("sampler must be 'rbf', 'nystroem' or None")
#     if train_test not in ['train', 'test']:
#         raise ValueError("train_test must be 'train' or 'test'")
#     m = model + "_"
#     s = "" if sampler is None else sampler + "_"
#     p = "" if not pca else "pca_"
#     b = "" if box_type is None else box_type + "_"
#     t =  train_test + " score"
#
#     r = m + s + p + b + t
#     return r

def get_params_from_models_bar(gui_dic, mod_bar):
    '''
    Returns a dictionary with the needed keys to pass to get_model_scores

    Parameters
    ----------
    mod_bar: a HBox with the widgets

    Return
    ------
    A dictionary with keys [model, dataset, features, sampler, pca, ensemble, box_type]
    '''
    # Todo Ahora mismo es funcional, pero es feo y poco seguro. Habrá que retocarlo
    '''
        hb = widgets.HBox([
        model_selector,
        sampler_selector,
        box_type_selector,
        n_estimators_selector,
        pca_checkbox,
    ])
    '''
    #dataset_name = dataset_selector.value
    dataset_name = gui_dic['dataset_selector'].value
    #di = get_data(dataset_name, n_ins = size_selector.value)
    di = get_data(dataset_name, n_ins = gui_dic['size_selector'].value)
    d = {
        'gui_dic': gui_dic,
        'model': mod_bar.children[0].value,
        'dataset': di,
        #'features':  None if mod_bar.children[1].value == "None" else np.linspace(*(features_selector.value), dtype = np.int64).tolist() if np.ediff1d(features_selector.value)[0] > 50 else np.arange(*(features_selector.value)).tolist(),
        'features':  None if mod_bar.children[1].value == "None" else np.linspace(*(gui_dic['features_selector'].value), dtype = np.int64).tolist() if np.ediff1d(gui_dic['features_selector'].value)[0] > 50 else np.arange(*(gui_dic['features_selector'].value)).tolist(),
        'sampler': mod_bar.children[1].value if mod_bar.children[1].value != "None" else None,
        'pca': mod_bar.children[4].value,
        'ensemble': None if mod_bar.children[1].value == "None" else mod_bar.children[3].value,
        # 'box_type': None if mod_bar.children[1].value == "None" else mod_bar.children[2].value if mod_bar.children[2].value != "None" else mod_bar.children[2].value,
        'box_type': None if mod_bar.children[1].value == "None" else mod_bar.children[2].value if mod_bar.children[2].value != "None" else None,
    }
    return d


def get_all_model_scores(gui_dic):
    '''
    Return two lists, one for all the train_dicts, and the other for all the test_dicts,
    based on the models bars in the GUI

    Returns
    -------
    A tuple with (test_dicts, train_dicts), where each one is a list with dictionarys
    '''
    train_dics = []
    test_dics = []
    #for c in models_bar.children:

    gui_dic['progress_bar'].min = 0
    gui_dic['progress_bar'].max = len(gui_dic['models_bar'].children)
    gui_dic['progress_bar'].value = 0
    for c in gui_dic['models_bar'].children:
        # c es un HBox
        train_dic, test_dic =  get_model_scores(**get_params_from_models_bar(gui_dic,c))
        train_dics.append(train_dic)
        test_dics.append(test_dic)
        gui_dic['progress_bar'].value += 1

    fig = plt.figure(figsize = (12.8,4.8))
    test_sp = fig.add_subplot(121)
    train_sp = fig.add_subplot(122, sharey = test_sp)


    test_sp.set_title("Test scores")
    train_sp.set_title("Train scores")
    fig.suptitle("{}, {} instances".format(gui_dic['dataset_selector'].value, gui_dic['size_selector'].value))
    for te, tr in zip(test_dics, train_dics):
        test_sp.plot(te['absi'], te['ord'], label = te['label'])
        train_sp.plot(tr['absi'], tr['ord'], label = tr['label'])
    test_sp.legend()
    train_sp.legend()
    plt.close()
    return fig

def get_model_scores(gui_dic,
              model,
              dataset,
              features = None,
              sampler = None,
              pca = False,
              ensemble = None,
              box_type = None):
    '''
    Parameters
    ----------
    model: string, 'dt', 'rf, 'linear_svc' or 'logit'
    dataset: dictionary with keys 'data_train', 'data_test', 'target_train', 'target_test'
    features: array with features to test or None
    sampler: string, 'rbf' or 'nystroem', or  None
    pca: bool
    ensemble: integer or None
    box_type: string, 'black', 'grey' or None (ignored if ensemble = None)

    Returns
    -------
    A tuple of two dictionarys, (train_dic,test_dic), each one with
    keys 'abs', 'ord', 'label'
    '''
    if not isinstance(features, (list, type(None))):
    # if not isinstance(features, (list)) and features != 'None':
        print("Features: {}".format(features))
        raise ValueError('features must be an ordered list of integers or None')

    # if features == 'None' and sampler != 'None':
    if features is None and sampler is not None:
        #print("Sampler: {}".format(sampler))
        raise ValueError('features is needed with sampler')

    data_train = dataset['data_train']
    data_test = dataset['data_test']
    target_train = dataset['target_train']
    target_test = dataset['target_test']

    if box_type is None:
        ensemble = None

    clf = get_model(model, sampler, pca, ensemble, box_type)
    if features is None:
        clf.fit(data_train, target_train)
        train_score = clf.score(data_train, target_train)
        test_score = clf.score(data_test, target_test)
        train_dic = {
            'absi': list(gui_dic['features_selector'].value), #los dos valores
            'ord': [train_score, train_score],
            'label': get_label(model, sampler, pca, box_type, 'train')
        }
        test_dic = {
            'absi': list(gui_dic['features_selector'].value),
            'ord': [test_score, test_score],
            'label': get_label(model, sampler, pca, box_type, 'test')
        }
    else:
        gui_dic['sub_progress_bar'].min = 0
        gui_dic['sub_progress_bar'].max = len(features)
        gui_dic['sub_progress_bar'].value = 0
        train_scores = []
        test_scores = []
        for f in features:
            clf.set_params(sampler__n_components = f)
            clf.fit(data_train, target_train)
            train_score = clf.score(data_train, target_train)
            test_score = clf.score(data_test, target_test)

            train_scores.append(train_score)
            test_scores.append(test_score)
            gui_dic['sub_progress_bar'].value += 1

        train_dic = {
            'absi': features,
            'ord': train_scores,
            'label': get_label(model, sampler, pca, box_type, 'train')
        }
        test_dic = {
            'absi': features,
            'ord': test_scores,
            'label': get_label(model, sampler, pca, box_type, 'test')
        }
    return train_dic, test_dic
