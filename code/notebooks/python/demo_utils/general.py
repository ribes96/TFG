import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

SUPPORTED_DATASETS = [
    "covertype",
    "digits",
    "fall_detection",
    # "mnist",
    "pen_digits",
    "satellite",
    "segment",
    "vowel",
]

# Commented for not using
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
#     t = train_test + " score"
#
#     r = m + s + p + b + t
#     return r


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
    # todo encargarse de mnist
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
