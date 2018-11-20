import pandas as pd
import numpy as np



def get_label(model, sampler, pca, box_type, train_test):
    '''Returns a string with the correct label
    Parameters
    ----------
    model: string, model name
    sampler: string, 'rbf' or 'nystroem', or  None
    pca: bool
    box_type: string, 'black', 'grey' or None
    train_test: string, 'train' or 'test'
    '''
    if sampler is not None and sampler not in ['rbf', 'nystroem']:
        raise ValueError("sampler must be 'rbf', 'nystroem' or None")
    if train_test not in ['train', 'test']:
        raise ValueError("train_test must be 'train' or 'test'")
    m = model + "_"
    s = "" if sampler is None else sampler + "_"
    p = "" if not pca else "pca_"
    b = "" if box_type is None else box_type + "_"
    t =  train_test + " score"

    r = m + s + p + b + t
    return r

def get_data(dataset_name, prop_train = 2/3, n_ins = None):
    '''
    Returns a dictionary with keys ['data_train', 'data_test', 'target_train', 'target_test'],
    with the data according to dataset name

    Parameters
    ----------
    dataset_name: str, name of the dataset. Possible values:
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
    prop_train: float, the proportion of instances in the train dataset, default: 2/3
    n_ins: get a subset of n_ins instances, if total number is smaller, is ignored
    Return
    ------
    A dictionary with the keys
    '''
    options = [
        "covertype",
        "digits",
        "fall_detection",
        "mnist",
        "pen_digits",
        "satellite",
        "segment",
        "vowel",
        ]
    if dataset_name not in options:
        raise ValueError("{} dataset is not available".format(dataset_name))
    if not 0 < prop_train < 1:
        raise ValueError("prop_train must be 0 < prop_train < 1")
    if not isinstance(n_ins, (type(None),int)):
        raise ValueError("Bad type for n_ins")
    if isinstance(n_ins, int) and n_ins <= 0:
        raise ValueError("n_ins must be positive")


    dir_path = '../../datasets/'
    file_path = '{0}{1}/{1}.csv'.format(dir_path, dataset_name)

    df = pd.read_csv(file_path)

    N = df.shape[0]
    if n_ins is not None:
        N = min(N, n_ins)
    N_train = np.ceil(N * prop_train).astype(np.int64)
    N_test = N - N_train

    data = df.drop(labels = 'target', axis = 1)
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
