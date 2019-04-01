from sklearn.preprocessing import FunctionTransformer
from sklearn.kernel_approximation import Nystroem
from sklearn.kernel_approximation import RBFSampler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.ensemble import BaggingClassifier
from sklearn.decomposition import PCA
import numpy as np

from scipy.linalg import LinAlgError


# TODO todas las demos (y funciones) que usan get_sampling_model_scores ahora
# darán error, porque ahora el retorno también incluye una lista de errores
# Es decir, la tupla ya no será de 2 elementos, sino de 3

# deprecating
# def get_base_model(model_name, C=1):
#     '''
#     Parameters
#     ----------
#     model_name : str
#         One of ['dt', 'linear_svc', 'logit']
#     C : num
#         Parameter for linear_svc. Ignored if model_name != 'linear_svc'
#
#     Returns
#     -------
#     Abstract model
#         Something on which you can call fit and score
#     '''
#     if model_name == 'dt':
#         m = DecisionTreeClassifier()
#     elif model_name == 'linear_svc':
#         # m = LinearSVC(C=1)
#         m = LinearSVC(C=C)
#     elif model_name == 'logit':
#         m = LogisticRegression(C=1, multi_class='multinomial',
# solver='lbfgs')
#     else:
#         raise ValueError('This model is not supported')
#
#     return m


# def get_sampler(sampler_name, gamma=0.2):
def get_sampler(sampler_name, rbfsampler_gamma, nystroem_gamma):
    '''
    Parameters
    ----------
    sampler_name : str
        One of ['identity', 'rbf', 'nystroem']

    Returns
    -------
    Transformer
        Something on which you can call fit and transform
    '''
    if sampler_name == 'identity':
        s = FunctionTransformer(None, validate=False)
    elif sampler_name == 'nystroem':
        s = Nystroem(gamma=nystroem_gamma)
    elif sampler_name == 'rbf':
        s = RBFSampler(gamma=rbfsampler_gamma)
    else:
        raise ValueError('This sampler ({}) is not supported'.format(sampler_name))
        # raise ValueError(f'This sampler ({sampler_name}) is not supported')
    return s


def get_pca(pca_bool):
    '''
    Parameters
    ----------
    pca_bool : bool
        Wheather to perform pca or not

    Returns
    -------
    Transformer
        Something on which you can call fit and transform
    '''
    if pca_bool:
        # p = PCA(n_components=0.9, svd_solver="full")
        p = PCA(n_components=0.95, svd_solver="full")
    else:
        p = FunctionTransformer(None, validate=False)
    return p


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


def get_oob(clf):
    '''
    Gets the oob of clf if it is possible, or None if it is not
    '''
    if isinstance(clf, BaggingClassifier):
        try:
            oob = clf.oob_score_
        except AttributeError:
            # Para cuando es un ensemble y no un bag
            oob = None
    elif isinstance(clf, Pipeline):
        # petará cuando haya sampling sin ensembling
        try:
            oob = clf.named_steps['model'].oob_score_
        except AttributeError:
            # Para cuando es un ensemble y no un bag
            oob = None
    return oob


def get_sampling_model_scores(clf, dataset, features, n_runs=10):
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
    (train, test, errors) : tuple of dict
        train and test are dict with keys ['ord', 'absi']
        errors
    '''
    # TODO quizá tocará actualizar la documentación respecto al retorno de
    # error

    data_train = dataset['data_train']
    data_test = dataset['data_test']
    target_train = dataset['target_train']
    target_test = dataset['target_test']

    train_scores = []
    test_scores = []
    oobs = []

    # temporal, quizá
    errors = []

    # Lo podría hacer dentro del for, pero siempre dará el mismo resultado
    try:
        clf.set_params(sampler__n_components=2)
        is_black = True
    except ValueError:
        clf.set_params(base_estimator__sampler__n_components=2)
        is_black = False

    for f in features:
        if is_black:
            clf.set_params(sampler__n_components=f)
        else:
            clf.set_params(base_estimator__sampler__n_components=f)

        oob = []
        train_score = []
        test_score = []
        found_error = False
        for _ in range(n_runs):
            try:
                clf.fit(data_train, target_train)
            except LinAlgError as lae:
                ddd = {
                    'clf': clf, 'data_train': data_train,
                    'target_train': target_train, 'error': lae,
                    'is_black': is_black, 'n_features': f}
                errors.append(ddd)
                found_error = True
                print("------- Ha habido un error")
                # continue
                break
            # TODO sucio, no me gusta nada
            oob.append(get_oob(clf))
            train_score.append(clf.score(data_train, target_train))
            test_score.append(clf.score(data_test, target_test))
        if found_error:
            continue
        train_score = np.mean(train_score)
        test_score = np.mean(test_score)
        try:
            oob = np.mean(oob)
        except TypeError:
            oob = None

        train_scores.append(train_score)
        test_scores.append(test_score)
        oobs.append(oob)

    # TODO Me parece cutre
    if oobs[0] is not None:
        train_dic = {
            'absi': features,
            'ord': train_scores,
            'oob': oobs,
        }
    else:
        train_dic = {
            'absi': features,
            'ord': train_scores,
        }
    test_dic = {
        'absi': features,
        'ord': test_scores,
    }
    return train_dic, test_dic, errors


###########################
# Rincón de pruebas
###########################
# Esta función recibe también un diccionario con los parámetros que necesita
# el base model. Las claves pueden variar, dependiendo de cual sea el
# base_model


# def get_model(model_name,
#               model_params,
#               rbfsampler_gamma,
#               nystroem_gamma,
#               sampler_name='identity',
#               pca_bool=False,
#               pca_first=False,
#               n_estim=None,
#               box_type='none'):
def get_model(model_name=None,
              model_params={},
              rbfsampler_gamma=None,
              nystroem_gamma=None,
              sampler_name=None,
              pca_bool=None,
              pca_first=None,
              n_estim=None,
              box_type=None):
    # No acepto parámetros por defecto, me los tienen que indicar todos
    '''
    Parameters
    ----------
    model_name : str
        One of One of ['dt', 'linear_svc', 'logit']
    model_params : dict
        Containing que parameters to use with the base_model
    sampler_name : str
        One of ['identity', 'rbf', 'nystroem']
    pca_bool : bool
        If pca is performed or not
    pca_first : bool
        If true, Pipeline is PCA >> Sampler >> Model. Else, is
        Sampler >> PCA >> Model
    n_estim : int or None
        n_estim > 0, ignored  if box_type == 'none'
    box_type : str
        One of ['black_bag', 'grey_bag', 'black_ens', 'grey_ens', 'none']
    gamma : float
        Just used when sampler_name in ['rbf', 'nystroem']. Parameter for those
        methods
    C : float
        Just used when model_name == 'linear_svc'. Parameter for that mehtod

    Returns
    -------
    An abstract model. Something to which you can call fit and score
    '''
    model = get_base_model_with_params(model_name=model_name,
                                       params=model_params)
    # sampler = get_sampler(sampler_name=sampler_name, gamma=gamma)
    sampler = get_sampler(sampler_name=sampler_name,
                          rbfsampler_gamma=rbfsampler_gamma,
                          nystroem_gamma=nystroem_gamma)
    pca = get_pca(pca_bool)

    # TODO a lo mejor se puede reducir código, pues hay mucho repetido
    if box_type == 'none':
        if pca_first:
            clf = Pipeline([
                ('pca', pca),
                ('sampler', sampler),
                ('model', model),
            ])
        else:
            clf = Pipeline([
                ('sampler', sampler),
                ('pca', pca),
                ('model', model),
            ])
    elif box_type == 'grey_bag':
        if pca_first:
            pipe = Pipeline([
                ('pca', pca),
                ('sampler', sampler),
                ('model', model),
            ])
        else:
            pipe = Pipeline([
                ('sampler', sampler),
                ('pca', pca),
                ('model', model),
            ])
        clf = BaggingClassifier(base_estimator=pipe, n_estimators=n_estim,
                                bootstrap=True, oob_score=True)
    elif box_type == 'black_bag':
        bag = BaggingClassifier(base_estimator=model, n_estimators=n_estim,
                                bootstrap=True, oob_score=True)
        if pca_first:
            clf = Pipeline([
                ('pca', pca),
                ('sampler', sampler),
                ('model', bag),
            ])
        else:
            clf = Pipeline([
                ('sampler', sampler),
                ('pca', pca),
                ('model', bag),
            ])
    elif box_type == 'grey_ens':
        if pca_first:
            pipe = Pipeline([
                ('pca', pca),
                ('sampler', sampler),
                ('model', model),
            ])
        else:
            pipe = Pipeline([
                ('sampler', sampler),
                ('pca', pca),
                ('model', model),
            ])
        clf = BaggingClassifier(base_estimator=pipe, n_estimators=n_estim,
                                bootstrap=False, oob_score=False)
    elif box_type == 'black_ens':
        bag = BaggingClassifier(base_estimator=model, n_estimators=n_estim,
                                bootstrap=False, oob_score=False)
        if pca_first:
            clf = Pipeline([
                ('pca', pca),
                ('sampler', sampler),
                ('model', bag),
            ])
        else:
            clf = Pipeline([
                ('sampler', sampler),
                ('pca', pca),
                ('model', bag),
            ])
    else:
        raise ValueError(f'This box_type ({box_type}) is not supported')

    return clf


def get_base_model_with_params(model_name, params):
    '''
    Parameters
    ----------
    model_name : str
        One of ['dt', 'linear_svc', 'logit', 'rbf_svc']
    params : dict
        Containing the parameters to use with the model creation

    Returns
    -------
    Abstract model
        Something on which you can call fit and score
    '''
    if model_name == 'dt':
        m = DecisionTreeClassifier(**params)
    elif model_name == 'linear_svc':
        # m = LinearSVC(C=1)
        # m = LinearSVC(**params, max_iter=5000)
        m = LinearSVC(**params, dual=False, tol=1e-2)
    elif model_name == 'logit':
        # m = LogisticRegression(**params,
        #                        multi_class='multinomial',
        #                        solver='lbfgs',
        #                        max_iter=1000)
        m = LogisticRegression(**params,
                               multi_class='multinomial',
                               solver='lbfgs',
                               tol=1e-2)
    elif model_name == 'rbf_svc':
        m = SVC(**params, kernel='rbf')
    else:
        raise ValueError('This model is not supported')

    return m
