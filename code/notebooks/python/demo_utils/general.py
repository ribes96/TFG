import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

SUPPORTED_DATASETS = [
    "segment",
    "covertype",
    "digits",
    "fall_detection",
    "mnist",
    "pen_digits",
    "satellite",
    "vowel",
]


def get_data(dataset_name, prop_train=2/3, n_ins=None):
    '''
    Returns a dictionary with tha specified dataset

    Parameters
    ----------
    dataset_name : str
        Name of the dataset. Possible values:
            [
            "covertype",
            "digits",
            "fall_detection",
            "mnist",
            "pen_digits",
            "satellite",
            "segment",
            "vowel",
            ]
    prop_train : float
        The proportion of instances in the train dataset, default: 2/3
    n_ins : int
        Get a subset of n_ins instances, if total number is smaller, is ignored
    Return
    ------
    dict
        With keys: ['data_train', 'data_test', 'target_train', 'target_test']
    '''
    options = SUPPORTED_DATASETS
    if dataset_name not in options:
        raise ValueError("{} dataset is not available".format(dataset_name))
    if not 0 < prop_train < 1:
        raise ValueError("prop_train must be 0 < prop_train < 1")

    dir_path = '../../datasets/'
    file_path = '{0}{1}/{1}.csv'.format(dir_path, dataset_name)

    df = pd.read_csv(file_path)

    N = df.shape[0]
    if n_ins is not None:
        N = min(N, n_ins)
    N_train = np.ceil(N * prop_train).astype(np.int64)
    # N_test = N - N_train  # commented for not using

    data = df.drop(labels='target', axis=1)
    target = df.target

    data_train = data.iloc[:N_train]
    data_test = data.iloc[N_train:N]

    target_train = target[:N_train]
    target_test = target[N_train:N]

    ret_dic = {
        'data_train': data_train,
        'data_test': data_test,
        'target_train': target_train,
        'target_test': target_test,
    }

    return ret_dic


def get_sampling_score_graph_from_scores(train_scores, test_scores):
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
    for te, tr in zip(test_scores, train_scores):
        test_sp.plot(te['absi'], te['ord'], label=te['label'])
        train_sp.plot(tr['absi'], tr['ord'], label=tr['label'])
    test_sp.legend()
    train_sp.legend()
    test_sp.grid(True)
    train_sp.grid(True)
    plt.close()
    return fig


def get_non_sampling_score_graph_from_scores(train_scores, test_scores):
    '''
    Returns a bar plot with the scores from parameters. You will be able to add
    a title to the returned object. Although it assumes non-sampling, the 'ord'
    must be a list with at least one element. Only the first one will be used

    Parameters
    ----------
    train_scores, test_scores : list of dict
        each dict with keys ['ord', 'label']. 'ord' must be a list although
        only the first element will be used

    Returns
    -------
    matplotlib.pyplot.Figure
        with two bar subplots, horizontally aligned, with test and train score,
        ready to be shown
    '''
    fig = plt.figure(figsize=(12.8, 4.8))
    test_sp = fig.add_subplot(121)
    train_sp = fig.add_subplot(122, sharey=test_sp)

    test_sp.set_title("Test scores")
    train_sp.set_title("Train scores")

    train_labels = [d['label'] for d in train_scores]
    test_labels = [d['label'] for d in test_scores]

    train_scores = [d['ord'][0] for d in train_scores]
    test_scores = [d['ord'][0] for d in test_scores]

    train_sp.bar(train_labels, train_scores)
    test_sp.bar(test_labels, test_scores)

    test_sp.grid(True)
    train_sp.grid(True)
    train_sp.set_xticklabels(train_labels, rotation=45)
    test_sp.set_xticklabels(test_labels, rotation=45)
    plt.close()
    return fig


def get_non_sampling_error_graph_from_scores(train_scores, test_scores):
    '''
    Returns a bar plot with the error from parameters. You will be able to add
    a title to the returned object. Although it assumes non-sampling, the 'ord'
    must be a list with at least one element. Only the first one will be used

    Parameters
    ----------
    train_scores, test_scores : list of dict
        each dict with keys ['ord', 'label']. 'ord' must be a list although
        only the first element will be used

    Returns
    -------
    matplotlib.pyplot.Figure
        with two bar subplots, horizontally aligned, with test and train error,
        ready to be shown
    '''
    fig = plt.figure(figsize=(12.8, 4.8))
    test_sp = fig.add_subplot(121)
    train_sp = fig.add_subplot(122, sharey=test_sp)

    test_sp.set_title("Test error")
    train_sp.set_title("Train error")

    train_labels = [d['label'] for d in train_scores]
    test_labels = [d['label'] for d in test_scores]

    train_scores = [1 - d['ord'][0] for d in train_scores]
    test_scores = [1 - d['ord'][0] for d in test_scores]

    train_sp.bar(train_labels, train_scores)
    test_sp.bar(test_labels, test_scores)

    test_sp.grid(True)
    train_sp.grid(True)
    train_sp.set_xticklabels(train_labels, rotation=45)
    test_sp.set_xticklabels(test_labels, rotation=45)
    plt.close()
    return fig


def get_sampling_error_graph_from_scores(train_scores, test_scores):
    '''
    Returns a graph with the errors from parameters. You will be able to add a
    title to the returned object

    Parameters
    ----------
    train_scores, test_scores : list of dict
        each dict with keys ['absi', 'ord', 'label']. 'absi' is required to
        be real units, -1 is not allowed

    Returns
    -------
    matplotlib.pyplot.Figure
        with two subplots, horizontally aligned, with test and train error,
        ready to be shown
    '''
    fig = plt.figure(figsize=(12.8, 4.8))
    test_sp = fig.add_subplot(121)
    train_sp = fig.add_subplot(122, sharey=test_sp)

    test_sp.set_title("Test error")
    train_sp.set_title("Train error")
    for te, tr in zip(test_scores, train_scores):
        test_error = [1 - i for i in te['ord']]
        train_error = [1 - i for i in tr['ord']]
        test_sp.plot(te['absi'], test_error, label=te['label'])
        train_sp.plot(tr['absi'], train_error, label=tr['label'])
    test_sp.legend()
    train_sp.legend()
    test_sp.grid(True)
    train_sp.grid(True)
    plt.close()
    return fig
