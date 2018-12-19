from sklearn.preprocessing import FunctionTransformer
from sklearn.kernel_approximation import Nystroem
from sklearn.kernel_approximation import RBFSampler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.ensemble import BaggingClassifier
from sklearn.decomposition import PCA

# TODO quitar la función get_model_first_pca (Demo5 la usa)


def get_base_model(model_name, C=1):
    '''
    Parameters
    ----------
    model_name : str
        One of ['dt', 'linear_svc', 'logit']
    C : num
        Parameter for linear_svc. Ignored if model_name != 'linear_svc'

    Returns
    -------
    Abstract model
        Something on which you can call fit and score
    '''
    if model_name == 'dt':
        m = DecisionTreeClassifier()
    elif model_name == 'linear_svc':
        # m = LinearSVC(C=1)
        m = LinearSVC(C=C)
    elif model_name == 'logit':
        m = LogisticRegression(C=1, multi_class='multinomial', solver='lbfgs')
    else:
        raise ValueError('This model is not supported')

    return m


def get_sampler(sampler_name, gamma=0.2):
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
        s = Nystroem(gamma=gamma)
    elif sampler_name == 'rbf':
        s = RBFSampler(gamma=gamma)
    else:
        raise ValueError('This sampler is not supported')
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
        p = PCA(n_components=0.9, svd_solver="full")
    else:
        p = FunctionTransformer(None, validate=False)
    return p


def get_model(model_name,
              sampler_name='identity',
              pca_bool=False,
              n_estim=None,
              box_type='none',
              gamma=0.2,
              C=1):
    '''
    Parameters
    ----------
    model_name : str
        One of One of ['dt', 'linear_svc', 'logit']
    sampler_name : str
        One of ['identity', 'rbf', 'nystroem']
    pca_bool : bool
    n_estim : int or None
        n_estim > 0, ignored  if box_type == 'none'
    box_type : str
        One of ['black', 'grey', 'none']
    gamma : float
        Just used when sampler_name in ['rbf', 'nystroem']. Parameter for those
        methods
    C : float
        Just used when model_name == 'linear_svc'. Parameter for that mehtod

    Returns
    -------
    An abstract model. Something to which you can call fit and score
    '''
    model = get_base_model(model_name=model_name, C=C)
    sampler = get_sampler(sampler_name=sampler_name, gamma=gamma)
    pca = get_pca(pca_bool)

    if box_type == 'none':
        clf = Pipeline([
            ('sampler', sampler),
            ('pca', pca),
            ('model', model),
        ])
    elif box_type == 'grey':
        pipe = Pipeline([
            ('sampler', sampler),
            ('pca', pca),
            ('model', model),
        ])
        clf = BaggingClassifier(base_estimator=pipe, n_estimators=n_estim,
                                oob_score=True)
    elif box_type == 'black':
        bag = BaggingClassifier(base_estimator=model, n_estimators=n_estim,
                                oob_score=True)
        clf = Pipeline([
            ('sampler', sampler),
            ('pca', pca),
            ('model', bag),
        ])
    else:
        raise ValueError('This box_type is not supported')

    return clf
############################

# TODO esta función es solo temporal
def get_model_first_pca(model_name,
                        sampler_name='identity',
                        pca_bool=False,
                        n_estim=None,
                        box_type='none',
                        gamma=0.2,
                        C=1):
    '''
    Parameters
    ----------
    model_name : str
        One of One of ['dt', 'linear_svc', 'logit']
    sampler_name : str
        One of ['identity', 'rbf', 'nystroem']
    pca_bool : bool
    n_estim : int or None
        n_estim > 0, ignored  if box_type == 'none'
    box_type : str
        One of ['black', 'grey', 'none']
    gamma : float
        Just used when sampler_name in ['rbf', 'nystroem']. Parameter for those
        methods
    C : float
        Just used when model_name == 'linear_svc'. Parameter for that mehtod

    Returns
    -------
    An abstract model. Something to which you can call fit and score
    '''
    model = get_base_model(model_name=model_name, C=C)
    sampler = get_sampler(sampler_name=sampler_name, gamma=gamma)
    pca = get_pca(pca_bool)

    if box_type == 'none':
        clf = Pipeline([
            ('pca', pca),
            ('sampler', sampler),
            ('model', model),
        ])
    elif box_type == 'grey':
        pipe = Pipeline([
            ('pca', pca),
            ('sampler', sampler),
            ('model', model),
        ])
        clf = BaggingClassifier(base_estimator=pipe, n_estimators=n_estim)
    elif box_type == 'black':
        bag = BaggingClassifier(base_estimator=model, n_estimators=n_estim)
        clf = Pipeline([
            ('pca', pca),
            ('sampler', sampler),
            ('model', bag),
        ])
    else:
        raise ValueError('This box_type is not supported')

    return clf

###############################


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
    oobs = []
    # Lo podría hacer dentro del for, pero siempre dará el mismo resultado
    try:
        clf.set_params(sampler__n_components=2)
        is_black = True
    except ValueError:
        clf.set_params(base_estimator__sampler__n_components=2)
        is_black = False

    for f in features:
        # try:
        #     clf.set_params(sampler__n_components=f)
        # except ValueError:
        #     clf.set_params(base_estimator__sampler__n_components=f)
        if is_black:
            #borrar
            # print('Es un black')
            # end borrar
            clf.set_params(sampler__n_components=f)
        else:
            # borrar
            # print('Es un grey')
            # end borrar
            clf.set_params(base_estimator__sampler__n_components=f)
        # borrar
        # print('El clf que usamos es:')
        # print(clf)
        # end borrar
        clf.fit(data_train, target_train)
        # TODO sucio, no me gusta nada
        if isinstance(clf, BaggingClassifier):
            try:
                oob = clf.oob_score_
            except AttributeError:
                # Para cuando es un ensemble y no un bag
                oob = None
        elif isinstance(clf, Pipeline):
            # TODO creo que cuando no sea un bagging debería petar,
            # pero no lo ha hecho
            # porque model, en vez de ser un bag, será un dt, por ejemplo
            # petará cuando haya sampling sin ensembling
            try:
                oob = clf.named_steps['model'].oob_score_
            except AttributeError:
                # Para cuando es un ensemble y no un bag
                oob = None
        train_score = clf.score(data_train, target_train)
        test_score = clf.score(data_test, target_test)

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
    return train_dic, test_dic


###########################
# Rincón de pruebas
###########################

# Esta función tiene que permitir más tipos de cajas, y permitir
# indicar el orden en el que quieres el pca
# Los nuevos tipos de cajas pasan a ser:
# black_bag
# grey_bag
# black_ens
# grey_ens

def get_model2(model_name,
               sampler_name='identity',
               pca_bool=False,
               pca_first=False,
               n_estim=None,
               box_type='none',
               gamma=0.2,
               C=1):
    '''
    Parameters
    ----------
    model_name : str
        One of One of ['dt', 'linear_svc', 'logit']
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
    model = get_base_model(model_name=model_name, C=C)
    sampler = get_sampler(sampler_name=sampler_name, gamma=gamma)
    pca = get_pca(pca_bool)

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
        raise ValueError('This box_type is not supported')

    return clf
