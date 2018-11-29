import pandas as pd
import numpy as np

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
    # todo la comprovación de parámetros creo que no habría que hacerla
    # todo encargarse de mnist
    options = SUPPORTED_DATASETS
    # options = [
    #     "covertype",
    #     "digits",
    #     "fall_detection",
    #     "mnist",
    #     "pen_digits",
    #     "satellite",
    #     "segment",
    #     "vowel",
    #     ]
    if dataset_name not in options:
        raise ValueError("{} dataset is not available".format(dataset_name))
    if not 0 < prop_train < 1:
        raise ValueError("prop_train must be 0 < prop_train < 1")
    if not isinstance(n_ins, (type(None), int)):
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

# Commented for not using
# def get_models_bar_data(models_bar):
#     '''
#     Returns a list of dictionarys, where each one represents one of the
#     models
#     to run, and which have keys ['model', 'sampling', 'box_type',
#     'n_estimators', 'pca']. The values are exactly the ones of the widget
#
#     Parameters
#     ----------
#     models_bar : VBox
#         containing many HBox, each with widgets [Dropdown,
#         Dropdown, Dropdown, IntSlider, Checkbox]
#
#     Returns
#     -------
#     list of dict
#         with keys ['model', 'sampling', 'box_type',
#         'n_estimators', 'pca']
#     '''
#     def get_single_model_bar_data(model_bar):
#         '''
#         For each element in the models_bar
#         Returns
#         -------
#         A dictionary
#         '''
#         d = {
#             'model': model_bar.children[0].value,
#             'sampling': model_bar.children[1].value,
#             'box_type': model_bar.children[2].value,
#             'n_estimators': model_bar.children[3].value,
#             'pca': model_bar.children[4].value,
#         }
#         return d
#     ret_list = [get_single_model_bar_data(bar) for bar in
#     models_bar.children]
#     return ret_list

# commented for not using
# def get_gui_data(gui_dic):
#     '''
#     Return a dictionary with static data about  the gui_dic
#
#     Currently it only returns the keys
#     ('dataset_selector', 'size_selector', 'features_selector')
#
#     Parameters
#     -----------
#     gui_dic: a dictionary with keys referencing items in gui, as returned by
#     function get_gui
#
#     Returns
#     -------
#     dict
#         with keys ['dataset_selector', 'size_selector',
#         'features_selector', 'models_bar']
#
#     '''
#     gui_data = {
#         'dataset_selector': gui_dic['dataset_selector'].value,
#         'size_selector': gui_dic['size_selector'].value,
#         'features_selector': gui_dic['features_selector'].value,
#
#         # 'models_bar':gui_dic['models_bar'],
#         'models_bar': get_models_bar_data(gui_dic['models_bar']),
#     }
#     return gui_data

    # gui_dic = {
    #     'calculate_bt': calculate_bt,
    #     'models_bar': models_bar,
    #     'dataset_selector': dataset_selector,
    #     'size_selector': size_selector,
    #     'features_selector': features_selector,
    #     'clear_output_button': clear_output_button,
    #     'progress_bar': progress_bar,
    #     'sub_progress_bar': sub_progress_bar,
    # }
