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
# import numpy as np

# from demo_utils.general import get_label
# from demo_utils.general import get_data


# todo esta función está hecha como el culo, y se utiliza en todas partes
# pensarla mejor
def get_model(model_name,
              sampler=None,
              pca=False,
              ensemble=None,
              box_type=None):
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
    clf : abstract model
        something on which you can call fit, score, and maybe
        set_params(sampler__n_components=f)
    a model with the parameters specified
    '''
    if model_name not in ['dt', 'linear_svc', 'logit']:
        raise ValueError('model {0} is not supported'.format(model_name))
    if sampler not in ['rbf', 'nystroem', None]:
        raise ValueError('sampler {0} is not supported'.format(sampler))
    if type(pca) != bool:
        raise ValueError('pca is a boolean')
    if type(ensemble) not in [int, type(None)]:
        raise ValueError('Wrong value for ensemble')
    if isinstance(ensemble, int) and ensemble < 1:
        raise ValueError('Number of estimators must be greater than 0')
    if box_type not in ['black', 'grey', None]:
        raise ValueError("box_type must be 'black', 'grey' or None")
    if box_type is not None and ensemble is None:
        raise ValueError("box_type doesn't match with ensemble")

    if sampler == 'rbf':
        s = RBFSampler(gamma=0.2)
    elif sampler == 'nystroem':
        s = Nystroem(gamma=0.2)
    elif sampler is None:
        s = FunctionTransformer(None, validate=False)
    else:
        # todo se supone que ya nos hemos asegurado antes
        # esto es solo para debugar
        raise ValueError("sampler is expected to be 'rbf', 'nystroem' or None")

    if pca:
        p = PCA(n_components=0.9, svd_solver="full")
    else:
        p = FunctionTransformer(None, validate=False)

    if model_name == 'dt':
        m = DecisionTreeClassifier()
    elif model_name == 'linear_svc':
        m = LinearSVC(C=1)
    elif model_name == 'logit':
        m = LogisticRegression(C=1, multi_class='multinomial', solver='lbfgs')

    if ensemble is None:
        clf = Pipeline([
            ('sampler', s),
            ('pca', p),
            ('model', m),
        ])
    elif box_type == 'black':
        bag = BaggingClassifier(base_estimator=m, n_estimators=ensemble)
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
        clf = BaggingClassifier(base_estimator=pipe, n_estimators=ensemble)

    return clf

# comented for not using
# def get_params_from_models_bar(gui_data, mod_bar):
#     '''
#     Returns a dictionary with the needed keys to pass to get_model_scores
#
#     Parameters
#     ----------
#     gui_data : dict
#         Required keys: ['dataset_selector', 'size_selector',
#         'features_selector']
#     mod_bar : dict
#         with keys ['model', 'sampling', 'box_type', 'n_estimators', 'pca'].
#         The values are directly values, not widgets
#
#     Return
#     ------
#     dict
#         With keys ['model', 'dataset', 'features', 'sampler', 'pca',
#         'ensemble', 'box_type']
#     '''
#
#     dataset_name = gui_data['dataset_selector']
#     di = get_data(dataset_name, n_ins=gui_data['size_selector'])
#     d = {
#         'gui_data': gui_data,
#         'model': mod_bar['model'],
#         'dataset': di,
#         'features': None if mod_bar['sampling'] == "None" else
#         np.linspace(*(gui_data['features_selector']),
#                     dtype=np.int64).tolist() if np.ediff1d(
#                         gui_data['features_selector']
#                     )[0] > 50 else np.arange(
#                         *(gui_data['features_selector'])
#                         ).tolist(),
#         'sampler': mod_bar['sampling'] if mod_bar['sampling'] != "None" else None,
#         'pca': mod_bar['pca'],
#         'ensemble': None if mod_bar['sampling'] == "None" else mod_bar['n_estimators'],
#         'box_type': None if mod_bar['sampling'] == "None" else mod_bar['box_type'] if mod_bar['box_type'] != "None" else None,
#     }
#     return d

# commented for not using
# def run_all_models(gui_data):
#     '''
#     Get the scores for all models
#
#     Returns two lists, one for train and one for test score. Each list contains
#     dictionarys, one for each model to train. The dictionarys have the tipical
#     keys ['absi', 'ord', 'label']
#
#     Parameters
#     -----------
#     gui_data : dict
#         Required keys: ['models_bar', 'dataset_selector', 'size_selector',
#         'features_selector']
#
#     Returns
#     -------
#     tuple of list of dict
#         (train_scores, test_scores), each dict with keys ['absi', 'ord',
#         'label']
#     '''
#     train_dics = []
#     test_dics = []
#     for c in gui_data['models_bar']:
#         # c es un HBox
#         train_dic, test_dic = get_model_scores(
#             **get_params_from_models_bar(gui_data, c)
#             )
#         train_dics.append(train_dic)
#         test_dics.append(test_dic)
#     return train_dics, test_dics


# todo no deberia estar en este fichero, pues no hace nada de learning
# def get_graph_from_scores(train_scores, test_scores, gui_data):
def get_graph_from_scores(train_scores, test_scores):
    '''
    Returns a graph with the scores from parameters. You will be able to add a
    title to the returned object

    Parameters
    ----------
    train_scores, test_scores : list of dict
        each dict with keys ['absi', 'ord', 'label']. 'absi' is required to
        be real units, -1 is not allowed

    Returns
    -------
    matplotlib.pyplot.Figure
        with two subplots, horizontally aligned, with test and train score,
        ready to be shown
    '''
    fig = plt.figure(figsize=(12.8, 4.8))
    test_sp = fig.add_subplot(121)
    train_sp = fig.add_subplot(122, sharey=test_sp)

    test_sp.set_title("Test scores")
    train_sp.set_title("Train scores")

    # fig.suptitle("{}, {} instances".format(
    #                             gui_data['dataset_selector'],
    #                             gui_data['size_selector']))
    for te, tr in zip(test_scores, train_scores):
        test_sp.plot(te['absi'], te['ord'], label=te['label'])
        train_sp.plot(tr['absi'], tr['ord'], label=tr['label'])
    test_sp.legend()
    train_sp.legend()
    test_sp.grid(True)
    train_sp.grid(True)
    plt.close()
    return fig


# commented for not using
# def get_all_model_scores(gui_data):
#     '''
#     Get the scores of the models in gui_data in the shape of a plt.figure
#
#     Parameters
#     ----------
#     gui_data : dict
#         Required keys: ['models_bar', 'dataset_selector', 'size_selector',
#         'features_selector']
#
#     Returns
#     -------
#     plt.figure
#         With two subplots in the same row, first for test scores, second for
#         train scores, ready to be shown
#     '''
#     train_scores, test_scores = run_all_models(gui_data)
#     # fig = get_graph_from_scores(train_scores, test_scores, gui_data)
#     fig = get_graph_from_scores(train_scores, test_scores)
#     return fig

# def __old_get_all_model_scores(gui_data):  # deprecating
#     '''
#     Return two lists, one for all the train_dicts, and the other for all the
#     test_dicts, based on the models bars in the GUI
#
#     Returns
#     -------
#     tuple of dict
#         with (test_dicts, train_dicts), where each one is a list with
#         dictionarys
#     '''
#     train_dics = []
#     test_dics = []
#     # for c in models_bar.children:
#
#     # gui_data['progress_bar'].min = 0
#     # gui_data['progress_bar'].max = len(gui_data['models_bar'].children)
#     # gui_data['progress_bar'].value = 0
#     # for c in gui_data['models_bar'].children:
#     for c in gui_data['models_bar']:
#         # c es un HBox
#         train_dic, test_dic = get_model_scores(
#             **get_params_from_models_bar(gui_data, c)
#             )
#         train_dics.append(train_dic)
#         test_dics.append(test_dic)
#
#         # gui_data['progress_bar'].value += 1
#
#     fig = plt.figure(figsize=(12.8, 4.8))
#     test_sp = fig.add_subplot(121)
#     train_sp = fig.add_subplot(122, sharey=test_sp)
#
#     test_sp.set_title("Test scores")
#     train_sp.set_title("Train scores")
#
#     # not value, directly
#     fig.suptitle("{}, {} instances".format(
#                                 gui_data['dataset_selector'],
#                                 gui_data['size_selector']))
#     for te, tr in zip(test_dics, train_dics):
#         test_sp.plot(te['absi'], te['ord'], label=te['label'])
#         train_sp.plot(tr['absi'], tr['ord'], label=tr['label'])
#     test_sp.legend()
#     train_sp.legend()
#     plt.close()
#     return fig


# def get_model_scores(gui_data,        # deprecating
#                      model,
#                      dataset,
#                      features=None,
#                      sampler=None,
#                      pca=False,
#                      ensemble=None,
#                      box_type=None):
#     '''
#     Parameters
#     ----------
#     model: string, 'dt', 'rf, 'linear_svc' or 'logit'
#     dataset: dictionary with keys 'data_train', 'data_test', 'target_train',
#     'target_test'
#     features: array with features to test or None
#     sampler: string, 'rbf' or 'nystroem', or  None
#     pca: bool
#     ensemble: integer or None
#     box_type: string, 'black', 'grey' or None (ignored if ensemble = None)
#
#     Returns
#     -------
#     A tuple of two dictionarys, (train_dic,test_dic), each one with
#     keys ['absi', 'ord', 'label']
#     '''
#     if not isinstance(features, (list, type(None))):
#         print("Features: {}".format(features))
#         raise ValueError('features must be an ordered list of\
#         integers or None')
#
#     # if features == 'None' and sampler != 'None':
#     if features is None and sampler is not None:
#         raise ValueError('features is needed with sampler')
#
#     data_train = dataset['data_train']
#     data_test = dataset['data_test']
#     target_train = dataset['target_train']
#     target_test = dataset['target_test']
#
#     if box_type is None:
#         ensemble = None
#
#     clf = get_model(model, sampler, pca, ensemble, box_type)
#     if features is None:
#         clf.fit(data_train, target_train)
#         train_score = clf.score(data_train, target_train)
#         test_score = clf.score(data_test, target_test)
#         train_dic = {
#             'absi': list(gui_data['features_selector']),  # los dos valores
#             'ord': [train_score, train_score],
#             'label': get_label(model, sampler, pca, box_type, 'train')
#         }
#         test_dic = {
#             # 'absi': list(gui_data['features_selector'].value),
#             'absi': list(gui_data['features_selector']),
#             'ord': [test_score, test_score],
#             'label': get_label(model, sampler, pca, box_type, 'test')
#         }
#     else:
#         # gui_data['sub_progress_bar'].min = 0
#         # gui_data['sub_progress_bar'].max = len(features)
#         # gui_data['sub_progress_bar'].value = 0
#         train_scores = []
#         test_scores = []
#         for f in features:
#             clf.set_params(sampler__n_components=f)
#             clf.fit(data_train, target_train)
#             train_score = clf.score(data_train, target_train)
#             test_score = clf.score(data_test, target_test)
#
#             train_scores.append(train_score)
#             test_scores.append(test_score)
#             # gui_data['sub_progress_bar'].value += 1
#
#         train_dic = {
#             'absi': features,
#             'ord': train_scores,
#             'label': get_label(model, sampler, pca, box_type, 'train')
#         }
#         test_dic = {
#             'absi': features,
#             'ord': test_scores,
#             'label': get_label(model, sampler, pca, box_type, 'test')
#         }
#     return train_dic, test_dic


def get_non_sampling_model_scores(clf, dataset):
    '''
    Assuming clf is a model which DOESN'T use sampling, get the scores for a
    given dataset

    Parameters
    ----------
    clf : abstract model
        needs to implement fit and score, like scikit-learn
    dataset : dict
        Required keys: ['data_train', 'data_test', 'target_train',
        'target_test']

    Returns
    -------
    tuple of float
        (train_score, test_score)
    '''
    data_train = dataset['data_train']
    data_test = dataset['data_test']
    target_train = dataset['target_train']
    target_test = dataset['target_test']

    clf.fit(data_train, target_train)

    train_score = clf.score(data_train, target_train)
    test_score = clf.score(data_test, target_test)

    return train_score, test_score


def get_sampling_model_scores(clf, dataset, features):
    '''
    Assuming clf is a model which DO use sampling, get the scores for a given
    dataset

    Parameters
    ----------
    clf : abstract model
        needs to implement set_params(sampler__n_components=f) and fit() and
        score(), like scikit-learn
    dataset : dict
        Required keys: ['data_train', 'data_test', 'target_train',
        'target_test']
    features : list of int
        The features on which to test

    Returns
    -------
    (train, test) : tuple of dict
        Dict with keys ['ord', 'absi']
    '''

    data_train = dataset['data_train']
    data_test = dataset['data_test']
    target_train = dataset['target_train']
    target_test = dataset['target_test']

    train_scores = []
    test_scores = []
    for f in features:
        clf.set_params(sampler__n_components=f)
        clf.fit(data_train, target_train)
        train_score = clf.score(data_train, target_train)
        test_score = clf.score(data_test, target_test)

        train_scores.append(train_score)
        test_scores.append(test_score)

    train_dic = {
        'absi': features,
        'ord': train_scores,
    }
    test_dic = {
        'absi': features,
        'ord': test_scores,
    }
    return train_dic, test_dic
