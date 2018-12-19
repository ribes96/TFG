from demo_utils.generic_demo import Demo
# from demo_utils.general import SUPPORTED_DATASETS
from demo_utils.learning import get_model
from demo_utils.learning import get_non_sampling_model_scores
from demo_utils.learning import get_sampling_model_scores
from demo_utils.general import get_data

import numpy as np
from ipywidgets import Button
from ipywidgets import Dropdown
from ipywidgets import RadioButtons
from ipywidgets import IntRangeSlider
from ipywidgets import VBox
from ipywidgets import HBox
from ipywidgets import Label
from ipywidgets import Layout
from ipywidgets import IntSlider
from ipywidgets import Checkbox


class Demo_deprec(Demo):
    desc = """# Una demo genérica"""

    def __init__(self):
        self.run_bt = Button(description='Demo_deprec', button_style='info')
        self.mod_add_bt = Button(description='Add model',
                                 button_style='info')
        self.mod_remove_bt = Button(description='Remove model',
                                    button_style='warning')
        # self.dts_selector = Dropdown(options=SUPPORTED_DATASETS,
        #                              value=SUPPORTED_DATASETS[0],
        #                              description='Dataset:')
        self.dts_selector = self.get_default_dts_selector()
        self.size_selector = RadioButtons(options=[1000, 2000, 5000, 10000],
                                          value=1000, description='Size')
        self.features_selector = IntRangeSlider(value=[30, 100], min=30,
                                                max=2000, step=1, layout=Layout(width='950px'))

        self.model_name_column = VBox([Label(value='Model')])
        self.sampler_name_column = VBox([Label(value='Sampler')])
        self.box_type_column = VBox([Label(value='Box Type')])
        self.n_estim_column = VBox([Label(value='Number Estimators')])
        self.pca_column = VBox([Label(value='PCA?')])

        self.models_bar = HBox([
            self.model_name_column,
            self.sampler_name_column,
            self.box_type_column,
            self.n_estim_column,
            self.pca_column,
        ], layout=Layout(border='3px solid black'))

        self.gui = VBox([
            self.dts_selector,
            self.size_selector,
            self.mod_add_bt,
            self.mod_remove_bt,
            self.features_selector,
            self.models_bar,
            self.run_bt,
        ])
        self.mod_add_bt.on_click(self.insert_model_bar)
        self.mod_remove_bt.on_click(self.remove_model_bar)
        self.insert_model_bar()
        # self.box_type_changed()
        self.sampler_changed()
        super().__init__()

    def models_gui_to_data(self, i):
        '''
        Parameters
        ----------
        i : int
            The position of the model to convert. The first one should be 1
        '''

        model_name = self.model_name_column.children[i].value
        sampler_name = self.sampler_name_column.children[i].value
        box_type = self.box_type_column.children[i].value
        n_estim = self.n_estim_column.children[i].value
        pca = self.pca_column.children[i].value

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

    def get_label(self, model_name, sampler_name, box_type, n_estim, pca):
        '''
        Parameters
        ----------
        model_name : str
            One of ['dt', 'logit', 'linear_svc']
        sampler_name : str
            One of ['rbf', 'nystroem', 'identity']
        box_type : str
            One of ['black', 'grey', 'none']
        n_estim : int or None
        pca : bool
        '''
        model_name += ' '
        if sampler_name == 'identity':
            sampler_name = ''
        else:
            sampler_name += ' '

        if box_type == 'none':
            box_type = ''
        else:
            box_type += ' '

        if n_estim is None:
            n_estim = ''
        else:
            n_estim = str(n_estim) + ' estims. '

        if pca:
            pca = 'PCA '
        else:
            pca = ''

        ret_str = model_name + sampler_name + box_type + n_estim + pca
        return ret_str

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
        info_run = '''
- Dataset: **{0}**
- Size: **{1}**
        '''
        self.run_specific = info_run.format(dts_name, dts_size)
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
                lab = self.get_label(model_name, sampler_name, box_type,
                                     n_estim, pca)
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
                lab = self.get_label(model_name, sampler_name, box_type,
                                     n_estim, pca)
                train_score['label'] = lab
                test_score['label'] = lab

            train_scores.append(train_score)
            test_scores.append(test_score)

        return train_scores, test_scores

    def insert_model_bar(self, e=None):
        la = Layout(width='90px')

        model_selector = Dropdown(
            options=['dt', 'logit', 'linear_svc'], value='dt', layout=la)
        sampler_selector = Dropdown(
            options=['None', 'rbf', 'nystroem'], value='None', layout=la)
        box_type_selector = Dropdown(
            options=['None', 'black', 'grey'], value='None', layout=la)
        n_estimators_selector = IntSlider(
            value=30, min=2, max=200,
            step=1,
            disabled=True,
            layout=Layout(width='300px', visibility='hidden'))
        pca_checkbox = Checkbox(value=False)

        self.model_name_column.children += (model_selector,)
        self.sampler_name_column.children += (sampler_selector,)
        self.box_type_column.children += (box_type_selector,)
        self.n_estim_column.children += (n_estimators_selector,)
        self.pca_column.children += (pca_checkbox, )

        def box_type_changed(*args):
            if box_type_selector.value == 'None':
                n_estimators_selector.disabled = True
                n_estimators_selector.layout.visibility = 'hidden'
            else:
                n_estimators_selector.disabled = False
                n_estimators_selector.layout.visibility = 'visible'

        box_type_selector.observe(box_type_changed, 'value')
        sampler_selector.observe(self.sampler_changed, 'value')

    def sampler_changed(self, *args):
        '''
        Sets features_selector weather enabled or disabled depending on
        the number of sampler == None in the gui
        '''
        val = True
        for i, row in enumerate(self.sampler_name_column.children):
            if i == 0:
                continue
            if row.value != 'None':
                val = False
                break
        if val:
            self.features_selector.disabled = True
            self.features_selector.layout.visibility = 'hidden'
        else:
            self.features_selector.disabled = False
            self.features_selector.layout.visibility = 'visible'

    def remove_model_bar(self, e=None):
        if len(self.models_bar.children[0].children) > 2:
            for i in range(5):
                self.models_bar.children[i].children =\
                    self.models_bar.children[i].children[:-1]
            self.sampler_changed()
