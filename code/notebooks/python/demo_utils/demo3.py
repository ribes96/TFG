from demo_utils.generic_demo import Demo
from demo_utils.general import SUPPORTED_DATASETS
# import ipywidgets as widgets
from demo_utils.learning import get_model
from demo_utils.general import get_data
from demo_utils.learning import get_sampling_model_scores
# from demo_utils.learning import get_non_sampling_model_scores
import numpy as np

from ipywidgets import Button
from ipywidgets import Dropdown
from ipywidgets import IntRangeSlider
from ipywidgets import VBox
from ipywidgets import IntSlider
from ipywidgets import Checkbox
# from ipywidgets import Layout


class Demo3(Demo):
    desc = '''# Diferencias entre los valores de gamma'''

    def __init__(self):
        self.gammas = [0.001, 0.01, 0.1, 1, 10, 100]
        self.all_datasets_names = SUPPORTED_DATASETS
        # TODO hardcoding
        self.n_ins = 1000
        self.run_bt = Button(description='Demo3', button_style='info')

        self.dataset_selector = Dropdown(options=SUPPORTED_DATASETS,
                                         value=SUPPORTED_DATASETS[0],
                                         description='Dataset:')
        self.model_selector = Dropdown(
            options=['dt', 'logit', 'linear_svc'], value='dt',
            description='Model')
        self.sampler_selector = Dropdown(
            options=['rbf', 'nystroem'], value='rbf', description='Sampler')
        self.features_selector = IntRangeSlider(value=[30, 100], min=30,
                                                max=400, step=10,
                                                description='Features')
        self.box_type_selector = Dropdown(options=['None', 'black', 'grey'],
                                          value='None',
                                          description='Bagging')
        self.n_estimators_selector = IntSlider(value=30, min=2, max=200,
                                               step=1, description='N. estim.')
        self.pca_checkbox = Checkbox(value=False, description='Perform PCA?')
        self.g = VBox([
            self.dataset_selector,
            self.model_selector,
            self.sampler_selector,
            self.features_selector,
            self.box_type_selector,
            self.n_estimators_selector,
            self.pca_checkbox,
        ])
        self.gui = VBox([self.g, self.run_bt])
        self.box_type_selector.observe(self.box_type_changed, 'value')
        # self.sampler_selector.observe(self.sampler_changed, 'value')
        # Solo para inhabilitar los que tocan
        self.box_type_changed()
        # self.sampler_changed()
        super().__init__()

    def gui_to_data(self):
        '''
        Just reading from self.gui, return a dictionary with keys and values
        needed to run the demo. Keys are the arguments of run_demo
        '''
        dts_name = self.dataset_selector.value
        model_name = self.model_selector.value
        sampler_name = self.sampler_selector.value
        box_type = self.box_type_selector.value
        if box_type == "None":
            box_type = 'none'
        n_estimators = self.n_estimators_selector.value
        if box_type == 'none':
            n_estimators = None
        pca = self.pca_checkbox.value
        features_range = self.features_selector.value

        model_data = {
            'model_name': model_name,
            'sampler_name': sampler_name,
            'pca_bool': pca,
            'n_estim': n_estimators,
            'box_type': box_type,
        }

        ret_dict = {
            'dts_name': dts_name,
            'model_data': model_data,
            'features_range': features_range,
        }
        return ret_dict

    def run_demo(self, dts_name, model_data, features_range):
        '''
        Parameters
        ----------
        dts_name : str
        model_data : dict
            Required keys: ['model_name', 'sampler_name', 'pca_bool',
            'n_estim', 'box_type']
            Almost everything get_model needs
        features_range : list
            The list is the range, so len(features) == 2, and increasing order
            is assumed
        '''
        info_run = '''
- Model: **{0}**
- Sampler: **{1}**
- Bagging: **{2}**
- N. estim.: **{3}**
- PCA: **{4}**
        '''
        self.run_specific = info_run.format(model_data['model_name'],
                                            model_data['sampler_name'],
                                            model_data['box_type'],
                                            model_data['n_estim'],
                                            model_data['pca_bool'])
        # a list of int is assumed
        n_splits_features = 30
        features = list(range(*features_range))
        if (features_range[1] - features_range[0]) > n_splits_features:
            features = np.linspace(*features_range, num=n_splits_features,
                                   dtype=np.int).tolist()
        return self.run_demo_with_sampling(dts_name, model_data, features)

    def run_demo_with_sampling(self, dts_name, model_data, features):
        '''
        Gets the score of many models, all equal (specified in model_data)
        except for the gamma value of the sampler, which uses self.gammas for
        each model.

        Parameters
        ----------
        dts_name : str
        model_data : dict
            Data needed to generate a model.
            Required keys: ['model_name', 'sampler_name', 'pca_bool',
            'n_estim', 'box_type']
        features : list of int
            A list with real features to test with. Values of -1 are not
            allowed

        Returns
        -------
        (train, test) : tuple of list of dict
            The scores for many models, which only disagree in the gamma value
            Keys of dict: ['absi', 'ord', 'label']
        '''
        train_dicts = []
        test_dicts = []
        dataset = get_data(dts_name, n_ins=self.n_ins)
        for g in self.gammas:
            model = get_model(gamma=g, **model_data)
            train_score, test_score = get_sampling_model_scores(model,
                                                                dataset,
                                                                features)
            train_score['label'] = 'gamma {}'.format(g)
            test_score['label'] = 'gamma {}'.format(g)

            train_dicts.append(train_score)
            test_dicts.append(test_score)

        return train_dicts, test_dicts

    def box_type_changed(self, *args):
        '''
        Desactiva n_estim_selector cuando no se hará bagging
        El parámetro *args es solo por ipywidgets, no me hace falta
        '''
        if self.box_type_selector.value == 'None':
            self.n_estimators_selector.disabled = True
            self.n_estimators_selector.layout.visibility = 'hidden'
        else:
            self.n_estimators_selector.disabled = False
            self.n_estimators_selector.layout.visibility = 'visible'
