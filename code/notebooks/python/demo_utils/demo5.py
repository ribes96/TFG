from demo_utils.generic_demo import Demo
from demo_utils.general import SUPPORTED_DATASETS
from demo_utils.general import get_data
from demo_utils.learning import get_model
from demo_utils.learning import get_model_first_pca
from demo_utils.learning import get_sampling_model_scores

from ipywidgets import Button
from ipywidgets import Dropdown
from ipywidgets import RadioButtons
from ipywidgets import IntRangeSlider
from ipywidgets import VBox
# from ipywidgets import HBox
# from ipywidgets import Label
# from ipywidgets import Layout
from ipywidgets import IntSlider
# from ipywidgets import Checkbox

import numpy as np


# TODO Hacer esta demo más ambiciosa y mostrar también el caso de no usar
# pca en absoluto, el modelo por sí solo y no estoy seguro de si hacer el
# modelo con pca sin sampler

# TODO que desaparezcan n_estim dependiendo de bagging

class Demo5(Demo):
    desc = '''# Orden adecuado entre Sampling y PCA
Para ver qué es mejor, primero hacer sampling y luego hacer PCA, o al revés
    '''

    def __init__(self):
        self.run_bt = Button(description='Demo5', button_style='info')
        self.dts_selector = Dropdown(options=SUPPORTED_DATASETS,
                                     value=SUPPORTED_DATASETS[0],
                                     description='Dataset:')
        self.size_selector = RadioButtons(options=[1000, 2000, 5000, 10000],
                                          value=1000, description='Size')
        self.model_selector = Dropdown(options=['dt', 'logit', 'linear_svc'],
                                       value='dt', description='Model')
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
        self.gui = VBox([
            self.dts_selector,
            self.model_selector,
            self.sampler_selector,
            self.features_selector,
            self.box_type_selector,
            self.n_estimators_selector,
            self.run_bt,
        ])
        super().__init__()

    def gui_to_data(self):
        '''
        Just reading from self.gui, return a dictionary with keys and values
        needed to run the demo. Keys are the arguments of run_demo

        Es importante que genere información, pero no estructuras abstractas
        que son resultado de llamar a algún método. Que contenga tipos de
        vanila python como numeros, strings, listas y diccionarios
        '''
        dts_name = self.dts_selector.value
        dts_size = self.size_selector.value
        model_name = self.model_selector.value
        sampler_name = self.sampler_selector.value
        box_type = self.box_type_selector.value
        if box_type == "None":
            box_type = 'none'
        n_estim = self.n_estimators_selector.value
        if box_type == 'none':
            n_estim = None
        features_range = features_range = self.features_selector.value
        model_data = {
            'model_name': model_name,
            'sampler_name': sampler_name,
            'n_estim': n_estim,
            'box_type': box_type,
        }

        ret_dict = {
            'dts_name': dts_name,
            'dts_size': dts_size,
            'model_data': model_data,
            'features_range': features_range,
        }
        return ret_dict

    def run_demo(self, dts_name, dts_size, model_data, features_range):
        '''
        Just reading from the arguments, return a pair of list of dictionarys,
        with the scores of the demo. Pair is (train, test)
        Parameters
        ----------
        dts_name : str
        dts_size : int
        model_data : dict
            Required keys: ['model_name', 'sampler_name', 'n_estim',
            'box_type']
            Almost everything get_model needs
        features_range : list
            The list is the range, so len(features) == 2, and increasing order
            is assumed
        '''
        # TODO hacer bien bien run_specific
        info_run = '''
- Dataset: **{0}**
- Size: **{1}**
- Model: **{2}**
- Sampler: **{3}**
- N. estim.: **{4}**
        '''
        self.run_specific = info_run.format(dts_name, dts_size,
                                            model_data['model_name'],
                                            model_data['sampler_name'],
                                            model_data['n_estim'])
        n_splits_features = 30
        features = list(range(*features_range))
        if (features_range[1] - features_range[0]) > n_splits_features:
            features = np.linspace(*features_range, num=n_splits_features,
                                   dtype=np.int).tolist()
        return self.run_demo_with_sampling(dts_name, dts_size, model_data,
                                           features)

    def run_demo_with_sampling(self, dts_name, dts_size, model_data, features):
        '''
        Gets the score of two models, (as specified in model_data)
        but one using first PCA and the other using first sampler

        Parameters
        ----------
        dts_name : str
        dts_size : int
        model_data : dict
            Data needed to generate a model.
            Required keys: ['model_name', 'sampler_name', 'n_estim',
            'box_type']
        features : list of int
            A list with real features to test with. Values of -1 are not
            allowed

        Returns
        -------
        (train, test) : tuple of list of dict
            The scores for two models specified, one first sampler, the other
            first pca
            Keys of dict: ['absi', 'ord', 'label']
        '''
        dataset = get_data(dts_name, n_ins=dts_size)
        model_first_sampler = get_model(pca_bool=True, **model_data)
        model_first_pca = get_model_first_pca(pca_bool=True, **model_data)

        first_sampler_train_score, first_sampler_test_score =\
            get_sampling_model_scores(model_first_sampler, dataset, features)
        first_sampler_train_score['label'] = 'Sampler >> PCA'
        first_sampler_test_score['label'] = 'Sampler >> PCA'

        first_pca_train_score, first_pca_test_score =\
            get_sampling_model_scores(model_first_pca, dataset, features)
        first_pca_train_score['label'] = 'PCA >> Sampler'
        first_pca_test_score['label'] = 'PCA >> Sampler'

        train_dicts = [first_sampler_train_score, first_pca_train_score]
        test_dicts = [first_sampler_test_score, first_pca_test_score]

        return train_dicts, test_dicts
