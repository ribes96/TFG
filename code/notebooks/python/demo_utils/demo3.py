from demo_utils.generic_demo import Demo
from demo_utils.general import SUPPORTED_DATASETS
# import ipywidgets as widgets
from demo_utils.learning import get_model
from demo_utils.general import get_data
from demo_utils.learning import get_sampling_model_scores
from demo_utils.learning import get_non_sampling_model_scores
import numpy as np

from ipywidgets import Button
from ipywidgets import Dropdown
# from ipywidgets import IntRangeSlider
from ipywidgets import VBox
from ipywidgets import Tab
# from ipywidgets import IntSlider
# from ipywidgets import Checkbox
from ipywidgets import Layout


class Demo3(Demo):
    desc = '''### Diferencias entre los valores de gamma'''

    def __init__(self):
        # self.gammas = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
        self.gammas = [10**i for i in range(-5, 3)]
        self.all_datasets_names = SUPPORTED_DATASETS
        self.run_bt = Button(description='Demo3', button_style='info')

        self.dataset_selector = self.get_default_dts_selector()
        self.size_selector = self.get_default_size_selector()
        self.dataset_selector.description = 'Dataset:'
        self.model_selector = self.get_default_model_selector()
        self.model_selector.description = 'Model'
        self.sampler_selector = Dropdown(
            options=['rbf', 'nystroem'], value='rbf', description='Sampler')
        self.features_selector = self.get_default_features_selector()
        self.features_selector.description = 'Features'
        self.features_selector.layout = Layout(width='800px')
        self.box_type_selector = self.get_default_box_type_selector()
        self.box_type_selector.description = 'Bagging'
        self.n_estimators_selector = self.get_default_n_estimators_selector()
        self.n_estimators_selector.description = 'N. estim.'
        self.pca_checkbox = self.get_default_pca_checkbox()
        self.pca_checkbox.description = 'Perform PCA?'
        self.pca_order_selector = self.get_default_pca_order_selector()
        self.g = VBox([
            self.dataset_selector,
            self.size_selector,
            self.model_selector,
            self.sampler_selector,
            self.features_selector,
            self.box_type_selector,
            self.n_estimators_selector,
            self.pca_checkbox,
            self.pca_order_selector,
        ])
        # self.gui = VBox([self.g, self.run_bt])
        self.tab0 = VBox([self.g, self.run_bt])
        self.tab1 = self.get_dt_hp_tab()
        self.tab2 = self.get_logit_hp_tab()
        self.tab3 = self.get_linearsvc_hp_tab()

        self.gui = Tab()
        self.gui.children = [self.tab0, self.tab1, self.tab2, self.tab3]
        tab_names = ['General', 'DT H-Params.', 'Logit H-Params.',
                     'SVC H-Params.']
        for i, e in enumerate(tab_names):
            self.gui.set_title(i, e)

        self.box_type_selector.observe(self.box_type_changed, 'value')
        self.pca_checkbox.observe(self.pca_checkbox_changed, 'value')
        # self.sampler_selector.observe(self.sampler_changed, 'value')
        # Solo para inhabilitar los que tocan
        self.box_type_changed()
        self.pca_checkbox_changed()
        # self.sampler_changed()
        super().__init__()

    def gui_to_data(self):
        '''
        Just reading from self.gui, return a dictionary with keys and values
        needed to run the demo. Keys are the arguments of run_demo
        '''
        dts_name = self.dataset_selector.value
        dts_size = self.size_selector.value
        model_name = self.model_selector.value
        sampler_name = self.sampler_selector.value
        box_type = self.box_type_selector.value
        if box_type == "None":
            box_type = 'none'
        n_estimators = self.n_estimators_selector.value
        if box_type == 'none':
            n_estimators = None
        pca = self.pca_checkbox.value
        pca_first = self.pca_order_selector.value
        features_range = self.features_selector.value

        model_data = {
            'model_name': model_name,
            'sampler_name': sampler_name,
            'pca_bool': pca,
            'pca_first': pca_first,
            'n_estim': n_estimators,
            'box_type': box_type,
        }
        dt_hp = {
            'max_depth': self.dt_max_depth_selector.value,
            'min_samples_split': self.dt_min_samples_split_selector.value,
            'min_samples_leaf': self.dt_min_samples_leaf_selector.value,
            'min_weight_fraction_leaf': self.dt_min_weight_fraction_leaf_selector.value,
            'max_leaf_nodes': int(self.dt_max_leaf_nodes_selector.value),
            'min_impurity_decrease': self.dt_min_impurity_decrease_selector.value,
        }
        logit_hp = {
            'C': self.logit_C_selector.value
        }
        linearsvc_hp = {
            'C': self.linearsvc_C_selector.value
        }

        ret_dict = {
            'dts_name': dts_name,
            'dts_size': dts_size,
            'model_data': model_data,
            'hparams': {
                'dt': dt_hp,
                'logit': logit_hp,
                # 'linearsvc': linearsvc_hp,
                'linear_svc': linearsvc_hp,
            },
            'features_range': features_range,
        }
        return ret_dict

    def run_demo(self, dts_name, dts_size, model_data, hparams, features_range):
        '''
        Parameters
        ----------
        dts_name : str
        model_data : dict
            Required keys: ['model_name', 'sampler_name', 'pca_bool',
            'n_estim', 'box_type']
            Almost everything get_model needs
        hparams: dict
            With keys ['dt', 'logit', 'linearsvc']
        features_range : list
            The list is the range, so len(features) == 2, and increasing order
            is assumed
        '''
#         info_run = '''
# - Model: **{0}**
# - Sampler: **{1}**
# - Bagging: **{2}**
# - N. estim.: **{3}**
# - PCA: **{4}**
#         '''
        info_run = {
            'Dataset': dts_name,
            'Size': dts_size,
            'Model': model_data['model_name'],
            'Sampler': model_data['sampler_name'],
            'Box': model_data['box_type'],
            'N. estimators': model_data['n_estim'],
            'PCA': model_data['pca_bool'],
            'PCA first': model_data['pca_first']
            # 'DT max. depth': hparams['dt']['max_depth'],
            # 'DT min. samples split': hparams['dt']['min_samples_split'],
            # 'DT min. samples leaf': hparams['dt']['min_samples_leaf'],
            # 'DT min. weight fraction leaf': hparams['dt']['min_weight_fraction_leaf'],
            # 'DT max. leaf nodes': hparams['dt']['max_leaf_nodes'],
            # 'DT min. impurity decrease': hparams['dt']['min_impurity_decrease'],
            # 'Logit C': hparams['logit']['C'],
            # 'Linear SVC': hparams['linearsvc']['C'],
        }
        dt_info_hparams = {
            'DT max. depth': hparams['dt']['max_depth'],
            'DT min. samples split': hparams['dt']['min_samples_split'],
            'DT min. samples leaf': hparams['dt']['min_samples_leaf'],
            'DT min. weight fraction leaf': hparams['dt']['min_weight_fraction_leaf'],
            'DT max. leaf nodes': hparams['dt']['max_leaf_nodes'],
            'DT min. impurity decrease': hparams['dt']['min_impurity_decrease'],
        }
        logit_info_hparams = {
            'Logit C': hparams['logit']['C'],
        }
        linearsvc_info_hparams = {
            # 'Linear SVC': hparams['linearsvc']['C'],
            'Linear SVC': hparams['linear_svc']['C'],
        }
        if model_data['model_name'] == 'dt':
            info_run.update(dt_info_hparams)
        elif model_data['model_name'] == 'logit':
            info_run.update(logit_info_hparams)
        elif model_data['model_name'] == 'linear_svc':
            info_run.update(linearsvc_info_hparams)
        else:
            print('Tenemos un problema')
        # self.run_specific = info_run.format(model_data['model_name'],
        #                                     model_data['sampler_name'],
        #                                     model_data['box_type'],
        #                                     model_data['n_estim'],
        #                                     model_data['pca_bool'])
        self.run_specific = self.get_run_specific_widget(info_run)
        self.train_scores.clear()
        self.test_scores.clear()
        # a list of int is assumed
        n_splits_features = 30
        features = list(range(*features_range))
        if (features_range[1] - features_range[0]) > n_splits_features:
            features = np.linspace(*features_range, num=n_splits_features,
                                   dtype=np.int).tolist()
        self.run_demo_with_sampling(dts_name=dts_name,
                                    dts_size=dts_size,
                                    model_data=model_data,
                                    hparams=hparams, features=features)

    def run_demo_with_sampling(self, dts_name, dts_size, model_data, hparams, features):
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
        dataset = get_data(dts_name, n_ins=dts_size)
        model_params = self.get_hparams(model_data['model_name'], hparams)
        for g in self.gammas:
            # model = get_model(gamma=g, **model_data)
            # TODO un poco cutre
            # model_params = self.get_hparams(model_data['model_name'],
            # hparams)
            # model_data tiene
            # 'model_name'
            # 'sampler_name'
            # 'pca_bool'
            # 'n_estim'
            # 'box_type'
            model = get_model(rbfsampler_gamma=g, nystroem_gamma=g,
                              model_params=model_params, **model_data)
            train_score, test_score, errors =\
                get_sampling_model_scores(model, dataset, features)
            train_score['label'] = 'gamma {}'.format(g)
            test_score['label'] = 'gamma {}'.format(g)

            train_dicts.append(train_score)
            test_dicts.append(test_score)

        # Ejecutar también el caso de no usar sampler
        # model_data['sampler_name'] = 'identity'
        m_data = dict(model_data)
        m_data['sampler_name'] = 'identity'
        model = get_model(rbfsampler_gamma=g, nystroem_gamma=g,
                          model_params=model_params, **m_data)
        tr_score, te_score = get_non_sampling_model_scores(model, dataset)
        train_score = {
            'absi': [features[0], features[-1]],
            'ord': [tr_score, tr_score],
            'label': 'No sampler'
        }
        test_score = {
            'absi': [features[0], features[-1]],
            'ord': [te_score, te_score],
            'label': 'No sampler'
        }
        # train_score['label'] = 'No sampler'
        # test_score['label'] = 'No sampler'

        train_dicts.append(train_score)
        test_dicts.append(test_score)

        # return train_dicts, test_dicts
        # self.train_scores.append(train_dicts)
        # self.test_scores.append(test_dicts)

        self.train_scores = train_dicts
        self.test_scores = test_dicts

    def get_hparams(self, model_name, hparams):
        '''
        Depending on the model name, return the dicionary inside hparams
        with the hyper-parameter
        '''
        if model_name == 'dt':
            return hparams['dt']
        if model_name == 'logit':
            return hparams['logit']
        if model_name == 'linear_svc':
            # return hparams['linearsvc']
            return hparams['linear_svc']
        raise ValueError('This model name is not supported')

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

    def pca_checkbox_changed(self, *args):
        '''
        Desactiva y hace invisible pca_order_selector cuando no se hará
        PCA
        '''
        if self.pca_checkbox.value:
            self.pca_order_selector.disabled = False
            self.pca_order_selector.layout.visibility = 'visible'
        else:
            self.pca_order_selector.disabled = True
            self.pca_order_selector.layout.visibility = 'hidden'
