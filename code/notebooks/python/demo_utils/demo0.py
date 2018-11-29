from demo_utils.generic_demo import Demo
import ipywidgets as widgets
from demo_utils.general import SUPPORTED_DATASETS
from demo_utils.learning import get_model
from demo_utils.learning import get_non_sampling_model_scores
from demo_utils.learning import get_sampling_model_scores
from demo_utils.general import get_data
import numpy as np


class Demo0(Demo):
    desc = """
    Una demo genérica, para comprobar cualquier cosa
    """
    run_bt = widgets.Button(description='Demo0', button_style='info')
    mod_add_bt = widgets.Button(description='Add model', button_style='info')
    mod_remove_bt = widgets.Button(description='Remove model',
                                   button_style='warning')
    dts_selector = widgets.Dropdown(options=SUPPORTED_DATASETS,
                                    value=SUPPORTED_DATASETS[0],
                                    descripttion='Dataset:')
    size_selector = widgets.RadioButtons(options=[1000, 2000, 5000, 10000],
                                         value=1000)
    # podría desconfigurarse el disabled si modifico get_new_model_bar
    features_selector = widgets.IntRangeSlider(value=[30, 100], min=30,
                                               max=400, step=10, disabled=True)

    models_bar = widgets.HBox([
        widgets.VBox([widgets.Label(value='Model')]),
        widgets.VBox([widgets.Label(value='Sampler')]),
        widgets.VBox([widgets.Label(value='Box Type')]),
        widgets.VBox([widgets.Label(value='Number Estimators')]),
        widgets.VBox([widgets.Label(value='PCA?')]),
    ], layout=widgets.Layout(border='3px solid black'))
    gui = widgets.VBox([
        dts_selector,
        size_selector,
        mod_add_bt,
        mod_remove_bt,
        features_selector,
        models_bar,
        run_bt,
    ])

    def __init__(self):
        self.mod_add_bt.on_click(self.insert_model_bar)
        self.mod_remove_bt.on_click(self.remove_model_bar)
        self.insert_model_bar()
        super().__init__()

    def models_gui_to_data(self, i):
        '''
        Parameters
        ----------
        i : int
            The position of the model to convert. The first one should be 1
        '''
        model_name = self.models_bar.children[0].children[i].value
        sampler_name = self.models_bar.children[1].children[i].value
        box_type = self.models_bar.children[2].children[i].value
        n_estim = self.models_bar.children[3].children[i].value
        pca = self.models_bar.children[4].children[i].value

        if sampler_name == 'None':
            sampler_name = 'identity'
        if box_type == 'None':
            box_type = 'none'

        if box_type == 'none':
            n_estim = None

        ret_dict = {
            'model_name': model_name,
            'sampler_name': sampler_name,
            'box_type': box_type,
            'n_estim': n_estim,
            'pca': pca,
        }
        return ret_dict

    def gui_to_data(self):
        '''
        Just reading from self.gui, return a dictionary with keys and values
        needed to run the demo. Keys are the arguments of run_demo

        Returns
        -------
        dict
            With keys: ['dts_name', 'dts_size', 'features_range', 'models']
        '''
        col = self.models_bar.children[0].children
        models = [self.models_gui_to_data(i) for i in range(1, len(col))]
        ret_dict = {
            'dts_name': self.dts_selector.value,
            'dts_size': self.size_selector.value,
            'features_range': self.features_selector.value,
            'models': models,
        }
        return ret_dict

    def run_demo(self, dts_name, dts_size, features_range, models):
        '''
        Just reading from the arguments, returns a pair of list of dictionarys,
        with the scores of the demo. Pair is (train, test)

        Parameters
        ----------
        dts_name : str
        dts_size : int
        features_range : list of int
            shape: [2], increasing order
        models : list of dict
            each dict is a model. Required keys: ['model_name', 'sampler',
            'box_type', 'n_estim', 'pca']
            Values of 'sampler' and 'box_type' are str or None

        Returns
        -------
        (train_scores, test_scores) : tuple of list of dict
            Dict with keys ['absi', 'ord', 'label']
        '''

        dataset = get_data(dts_name, n_ins=dts_size)
        train_scores = []
        test_scores = []

        for m in models:
            model_name = m['model_name']
            sampler_name = m['sampler_name']
            box_type = m['box_type']
            n_estim = m['n_estim']
            pca = m['pca']
            if box_type == 'none':
                n_estim = None
            clf = get_model(model_name=model_name,
                            sampler_name=sampler_name,
                            pca_bool=pca,
                            n_estim=n_estim,
                            box_type=box_type)
            n_splits_features = 30
            features = list(range(*features_range))
            if (features_range[1] - features_range[0]) > n_splits_features:
                features = np.linspace(*features_range,
                                       num=n_splits_features,
                                       dtype=np.int).tolist()

            if sampler_name == 'identity':
                features = None

            if sampler_name is 'identity':
                # train_score y test_score son floats
                train_score, test_score =\
                    get_non_sampling_model_scores(clf, dataset)
                # todo pensar el label
                lab = '{0}_{1}_{2}_{3}_{4}'.format(model_name, sampler_name,
                                                   box_type, n_estim, pca)
                train_score = {
                    'absi': features_range,
                    'ord': [train_score, train_score],
                    'label': lab,
                }
                test_score = {
                    'absi': features_range,
                    'ord': [test_score, test_score],
                    'label': lab,
                }
            else:
                # train_score y test_score son diccionarios
                train_score, test_score =\
                    get_sampling_model_scores(clf, dataset, features)
                # todo pensar el label
                lab = '{0}_{1}_{2}_{3}_{4}'.format(model_name, sampler_name,
                                                   box_type, n_estim, pca)
                train_score['label'] = lab
                test_score['label'] = lab

            train_scores.append(train_score)
            test_scores.append(test_score)

        return train_scores, test_scores

    def widget_set_to_data(self, model_tuple):
        '''
        Reads the widgets of a model and returns its data in a dict

        Parameters
        ----------
        model_tuple : iterable
            Any iterable containg the widgets of a model
            It is assumed:
                model_tupe[0] : Dropdown -> model_name
                model_tupe[1] : Dropdown -> sampler
                model_tupe[2] : Dropdown -> box_type
                model_tupe[3] : IntRangeSlider -> n_estim
                model_tupe[4] : Checkbox -> pca

        Returns
        -------
        dict
            With keys: ['model_name', 'sampler', 'box_type', 'n_estim', 'pca']
        '''
        model_name = model_tuple[0].value
        sampler = model_tuple[1].value
        box_type = model_tuple[2].value
        n_estim = model_tuple[3].value
        pca = model_tuple[4].value

        if sampler == 'None':
            sampler = 'identity'
        if box_type == 'None':
            box_type = 'none'

        if box_type == 'none':
            n_estim = None

        ret_dict = {
            'model_name': model_name,
            'sampler_name': sampler,
            'box_type': box_type,
            'n_estim': n_estim,
            'pca': pca,
        }
        return ret_dict

    def insert_model_bar(self, e=None):
        la = widgets.Layout(width='90px')

        model_selector = widgets.Dropdown(
            options=['dt', 'logit', 'linear_svc'], value='dt', layout=la)
        sampler_selector = widgets.Dropdown(
            options=['None', 'rbf', 'nystroem'], value='None', layout=la)
        box_type_selector = widgets.Dropdown(
            options=['None', 'black', 'grey'], value='None', layout=la)
        n_estimators_selector = widgets.IntSlider(
            value=30, min=2, max=200,
            step=1,
            disabled=True,
            layout=widgets.Layout(width='300px'))
        pca_checkbox = widgets.Checkbox(value=False)

        self.models_bar.children[0].children += (model_selector,)
        self.models_bar.children[1].children += (sampler_selector,)
        self.models_bar.children[2].children += (box_type_selector,)
        self.models_bar.children[3].children += (n_estimators_selector,)
        self.models_bar.children[4].children += (pca_checkbox, )

        def box_type_changed(*args):
            if box_type_selector.value == 'None':
                n_estimators_selector.disabled = True
            else:
                n_estimators_selector.disabled = False

        def sampler_changed(*args):
            val = True
            # todo hardcodeamos 1
            col = self.models_bar.children[1]
            for i, row in enumerate(col.children):
                if i == 0:
                    continue
                if row.value != 'None':
                    val = False
                    break
            if val:
                self.features_selector.disabled = True
            else:
                self.features_selector.disabled = False

        box_type_selector.observe(box_type_changed, 'value')
        sampler_selector.observe(sampler_changed, 'value')

    def remove_model_bar(self, e=None):
        if len(self.models_bar.children[0].children) > 2:
            for i in range(5):
                self.models_bar.children[i].children =\
                    self.models_bar.children[i].children[:-1]
