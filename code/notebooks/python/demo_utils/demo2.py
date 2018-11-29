from demo_utils.generic_demo import Demo
from demo_utils.general import SUPPORTED_DATASETS
import ipywidgets as widgets
from demo_utils.learning import get_model
from demo_utils.general import get_data
from demo_utils.learning import get_sampling_model_scores
from demo_utils.learning import get_non_sampling_model_scores
import numpy as np


class Demo2(Demo):
    desc = '''
    Permite ver si con datasets distintos los resultados son distintos.
    Prueba un modelo determinado con todos los datasets disponibles.
    '''
    all_datasets_names = SUPPORTED_DATASETS
    run_bt = widgets.Button(description='Demo2', button_style='info')

    model_selector = widgets.Dropdown(
        options=['dt', 'logit', 'linear_svc'], value='dt', description='Model')
    sampler_selector = widgets.Dropdown(
        options=['None', 'rbf', 'nystroem'], value='None',
        description='Sampler')
    features_selector = widgets.IntRangeSlider(value=[30, 100], min=30,
                                               max=400, step=10, disabled=True,
                                               description='Features')
    box_type_selector = widgets.Dropdown(
        options=['None', 'black', 'grey'], value='None', description='Bagging')
    n_estimators_selector = widgets.IntSlider(
        value=30, min=2, max=200, step=1,
        disabled=True, description='N. estim.')
    pca_checkbox = widgets.Checkbox(value=False, description='Perform PCA?')
    g = widgets.VBox([
        model_selector,
        sampler_selector,
        features_selector,
        box_type_selector,
        n_estimators_selector,
        pca_checkbox,
    ])
    gui = widgets.VBox([g, run_bt])

    def __init__(self):
        self.box_type_selector.observe(self.box_type_changed, 'value')
        self.sampler_selector.observe(self.sampler_changed, 'value')
        super().__init__()

    def gui_to_data(self):
        '''
        Just reading from self.gui, return a dictionary with keys and values
        needed to run the demo. Keys are the arguments of run_demo
        '''
        model_name = self.model_selector.value
        sampler_name = self.sampler_selector.value
        if sampler_name == "None":
            sampler_name = 'identity'
        box_type = self.box_type_selector.value
        if box_type == "None":
            box_type = 'none'
        n_estimators = self.n_estimators_selector.value
        if box_type == 'none':
            n_estimators = None
        pca = self.pca_checkbox.value
        features_range = self.features_selector.value

        clf = get_model(model_name=model_name,
                        sampler_name=sampler_name,
                        pca_bool=pca,
                        n_estim=n_estimators,
                        box_type=box_type)

        if sampler_name == 'identity':
            features_range = None
        ret_dict = {
            'model': clf,
            'features_range': features_range,
        }
        return ret_dict

    def run_demo(self, model, features_range):
        '''
        Parameters
        ----------
        model : abstract model
        features_range : list or None
            The list is the range, so len(features) == 2, and increasing order
            is assumed
        '''
        if features_range is None:
            return self.run_demo_non_sampling(model)
        else:
            # a list of int is assumed
            n_splits_features = 30
            features = list(range(*features_range))
            if (features_range[1] - features_range[0]) > n_splits_features:
                features = np.linspace(*features_range, num=n_splits_features,
                                       dtype=np.int).tolist()
            return self.run_demo_with_sampling(model, features)

    def run_demo_with_sampling(self, model, features):
        train_dicts = []
        test_dicts = []
        for dts_name in self.all_datasets_names:
            dataset = get_data(dts_name)
            train_score, test_score = get_sampling_model_scores(model,
                                                                dataset,
                                                                features)
            train_score['label'] = dts_name
            test_score['label'] = dts_name

            train_dicts.append(train_score)
            test_dicts.append(test_score)

        return train_dicts, test_dicts

    def run_demo_non_sampling(self, model):
        # run_demo llamará a esta o a la otra dependiendo del tipo.
        '''
        Parameters
        ----------
        model : abstract model
            Something on which you can call fit and score

        Returns
        -------
        (train_scores, test_scores) : tuple of list of dict
            dict with keys ['absi', 'ord', 'labels']
        '''
        train_scores = []
        test_scores = []
        for dts_name in self.all_datasets_names:
            dataset = get_data(dts_name)
            train_score, test_score = get_non_sampling_model_scores(model,
                                                                    dataset)
            train_scores.append(train_score)
            test_scores.append(test_score)
        train_dicts = []
        test_dicts = []
        for i, dts_name in enumerate(self.all_datasets_names):
            train_d = {
                'absi': [0, 1],
                'ord': [train_scores[i], train_scores[i]],
                'label': dts_name,
            }
            test_d = {
                'absi': [0, 1],
                'ord': [test_scores[i], test_scores[i]],
                'label': dts_name,
            }
            train_dicts.append(train_d)
            test_dicts.append(test_d)

        return train_dicts, test_dicts

    def box_type_changed(self, *args):
        '''
        Desactiva n_estim_selector cuando no se hará bagging
        El parámetro *args es solo por ipywidgets, no me hace falta
        '''
        if self.box_type_selector.value == 'None':
            self.n_estimators_selector.disabled = True
        else:
            self.n_estimators_selector.disabled = False

    def sampler_changed(self, *args):
        '''
        Desactiva features_selector cuando no se hará bagging
        El parámetro *args es solo por ipywidgets, no me hace falta
        '''
        if self.sampler_selector.value == 'None':
            self.features_selector.disabled = True
        else:
            self.features_selector.disabled = False
