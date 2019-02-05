from demo_utils.generic_demo import Demo
# from demo_utils.general import SUPPORTED_DATASETS
from demo_utils.learning import get_model
# from demo_utils.learning import get_model_with_params
from demo_utils.learning import get_non_sampling_model_scores
from demo_utils.learning import get_sampling_model_scores
from demo_utils.general import get_data

import numpy as np
# import ipywidgets as widgets
from ipywidgets import Button
# from ipywidgets import Dropdown
# from ipywidgets import RadioButtons
# from ipywidgets import IntRangeSlider
from ipywidgets import VBox
# from ipywidgets import FloatLogSlider
# from ipywidgets import IntSlider
# from ipywidgets import FloatSlider
from ipywidgets import HBox
from ipywidgets import Label
from ipywidgets import Layout
from ipywidgets import Tab
# from ipywidgets import HTML
# from ipywidgets import Accordion
# from ipywidgets import IntSlider
# from ipywidgets import Checkbox

# TODO una demo para ver cómo no sobreajustar DT es peor

# TODO una demo para ver el orden adecuado para pca y sampler

# TODO solucionar exportar que saque gráficas

# TODO poner un botón a cada modelo para quitar solamente ese

# TODO quizá se pueden añadir más modelos

# TODO poner quizá un acordeón con opciones avanzadas como la gamma y a C
# De hecho, podría contener los parámetros para cada uno de los modelos

# TODO ahora mismo las gráficas se generan, se muestran y se olvidan, no se
# guardan. Que las gráficas (o mejor, la información que genera las gráficas)
# se guarden en algún sitio de la clase para que se puedan consultar luego

# TODO todas las demos que llaman a get_model, que especifiquen el orden
# que quieren con PCA

# TODO poner un método a la demo0 que te muestre un texto con lo que
# tienes que poner para ejecutar la demo especificada en la gui de modo
# no interactivo, de modo que solo haya que hacer copy paste


class Demo10(Demo):
    desc = """### Demo genérica v10"""

    def __init__(self):
        self.run_bt = Button(description='Demo10', button_style='info')
        self.mod_add_bt = Button(description='Add model',
                                 button_style='info')
        self.mod_remove_bt = Button(description='Remove model',
                                    button_style='warning')
        self.dts_selector = self.get_default_dts_selector()
        self.dts_selector.description = 'Dataset:'
        self.size_selector = self.get_default_size_selector()
        self.features_selector = self.get_default_features_selector()
        self.features_selector.layout = Layout(width='900px')
        self.model_name_column = VBox([Label(value='Model')])
        self.sampler_name_column = VBox([Label(value='Sampler')])
        self.box_type_column = VBox([Label(value='Box Type')])
        self.n_estim_column = VBox([Label(value='Number Estimators')])
        self.pca_column = VBox([Label(value='PCA?')])
        self.pca_order_column = VBox([Label(value='PCA Order')])

        self.models_bar = HBox([
            self.model_name_column,
            self.sampler_name_column,
            self.box_type_column,
            self.n_estim_column,
            self.pca_column,
            self.pca_order_column,
        ], layout=Layout(border='3px solid black'))

        self.gui = Tab()

        self.tab0 = VBox([
            self.dts_selector,
            self.size_selector,
            self.mod_add_bt,
            self.mod_remove_bt,
            self.features_selector,
            self.models_bar,
            self.run_bt,
        ])

        self.tab1 = self.get_dt_hp_tab()
        self.tab2 = self.get_logit_hp_tab()
        self.tab3 = self.get_linearsvc_hp_tab()
        self.tab4 = self.get_samplers_params_tab()

        self.gui.children = [self.tab0, self.tab1, self.tab2, self.tab3,
                             self.tab4]
        tab_names = ['General', 'DT H-Params.', 'Logit H-Params.',
                     'SVC H-Params.', 'Samplers Params']
        for i, e in enumerate(tab_names):
            self.gui.set_title(i, e)
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
        pca_first = self.pca_order_column.children[i].value

        if sampler_name == 'None':
            sampler_name = 'identity'
        # TODO quizá habrá que cambiar los otros
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
            'pca_first': pca_first,
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
        # TODO actualizar la documentación
        col = self.models_bar.children[0].children
        models = [self.models_gui_to_data(i) for i in range(1, len(col))]
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
            'dts_name': self.dts_selector.value,
            'dts_size': self.size_selector.value,
            'features_range': self.features_selector.value,
            'models': models,
            'rbfsampler_gamma': self.rbf_gamma_selector.value,
            'nystroem_gamma': self.nystroem_gamma_selector.value,
            'hparams': {
                'dt': dt_hp,
                'logit': logit_hp,
                # 'linearsvc': linearsvc_hp,
                'linear_svc': linearsvc_hp,
            }
        }
        return ret_dict

    def get_label(self, model_name, sampler_name, box_type, n_estim, pca, pca_first):
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
            Weather to perform pca
        pca_first : bool
            Weather pca or model is first
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
            if pca_first:
                pca = 'PCA (first)'
            else:
                pca = 'PCA (last)'
            # pca = 'PCA '
        else:
            pca = ''

        ret_str = model_name + sampler_name + box_type + n_estim + pca
        # ret_str += str(pca_first)
        return ret_str

    def insert_model_bar(self, e=None):
        # just for not repeating
        DD_lay = Layout(width='90px', margin='8px 2px 8px 2px')

        model_selector = self.get_default_model_selector()
        model_selector.layout = DD_lay
        sampler_selector = self.get_default_sampler_selector()
        sampler_selector.layout = DD_lay
        box_type_selector = self.get_default_box_type_selector()
        box_type_selector.layout = DD_lay
        n_estimators_selector = self.get_default_n_estimators_selector()
        n_estimators_selector.layout = Layout(width='300px',
                                              visibility='hidden',
                                              margin='8px 2px 8px 2px')
        # n_estimators_selector.disabled = True
        pca_checkbox = self.get_default_pca_checkbox()
        pca_checkbox.layout = Layout(width='50px', margin='8px 2px 8px 2px')
        pca_order_selector = self.get_default_pca_order_selector()
        pca_order_selector.layout = Layout(width='150px', visibility='hidden')
        # pca_order_selector.disabled = True

        self.model_name_column.children += (model_selector,)
        self.sampler_name_column.children += (sampler_selector,)
        self.box_type_column.children += (box_type_selector,)
        self.n_estim_column.children += (n_estimators_selector,)
        self.pca_column.children += (pca_checkbox, )
        self.pca_order_column.children += (pca_order_selector, )

        def box_type_changed(*args):
            if box_type_selector.value == 'None':
                n_estimators_selector.disabled = True
                n_estimators_selector.layout.visibility = 'hidden'
            else:
                n_estimators_selector.disabled = False
                n_estimators_selector.layout.visibility = 'visible'

        box_type_selector.observe(box_type_changed, 'value')
        sampler_selector.observe(self.sampler_changed, 'value')

        def pca_changed(*args):
            if pca_checkbox.value:
                pca_order_selector.disabled = False
                pca_order_selector.layout.visibility = 'visible'
            else:
                pca_order_selector.disabled = True
                pca_order_selector.layout.visibility = 'hidden'

        pca_checkbox.observe(pca_changed, 'value')

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
            # TODO este hardcoding quizá se puede evitar
            for i in range(6):
                self.models_bar.children[i].children =\
                    self.models_bar.children[i].children[:-1]
            self.sampler_changed()

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

    # def get_run_specific_widget(self, d):
    #     '''
    #     Returns a widget (accordion) full of labels showing the info in d, with
    #     the key in bold
    #     '''
    #     labels = [HTML(value='<strong>{0}</strong>: {1}'.format(k, d[k])) for k in d]
    #     v = VBox(labels)
    #     ac = Accordion([v])
    #     ac.set_title(0, 'Info run')
    #     return ac

    def run_demo(self, dts_name, dts_size, features_range, models, hparams,
                 rbfsampler_gamma, nystroem_gamma):
        '''
        First it clears self.train_scores and self.test_scores, and then
        runs the demo, appending to those the results of each of the models.

        The results are in the shape of a dictionary, with keys ['absi', 'ord',
        'label']

        It is resistant to failing of some of the models. If that happens, a
        warning in raised, self.ERRORS is filled with some info, and the execution
        continues with the rest of the models.

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
        hparams : dict
            Required keys: ['dt', 'logit', 'linearsvc']

        Returns
        -------
        None
        '''
        info_run = '''
- Dataset: **{0}**
- Size: **{1}**
        '''
        # self.run_specific = info_run.format(dts_name, dts_size)

        info_run = {
            'Dataset': dts_name,
            'Size': dts_size,
            'RFF gamma': rbfsampler_gamma,
            'Nystroem gamma': nystroem_gamma,
            'DT max. depth': hparams['dt']['max_depth'],
            'DT min. samples split': hparams['dt']['min_samples_split'],
            'DT min. samples leaf': hparams['dt']['min_samples_leaf'],
            'DT min. weight fraction leaf': hparams['dt']['min_weight_fraction_leaf'],
            'DT max. leaf nodes': hparams['dt']['max_leaf_nodes'],
            'DT min. impurity decrease': hparams['dt']['min_impurity_decrease'],
            'Logit C': hparams['logit']['C'],
            # 'Linear SVC': hparams['linearsvc']['C'],
            'Linear SVC': hparams['linear_svc']['C'],
        }
        self.run_specific = self.get_run_specific_widget(info_run)
        dataset = get_data(dts_name, n_ins=dts_size)
        # train_scores = []
        # test_scores = []

        self.train_scores.clear()
        self.test_scores.clear()

        for m in models:
            model_name = m['model_name']
            sampler_name = m['sampler_name']
            box_type = m['box_type']
            n_estim = m['n_estim']
            pca = m['pca']
            pca_first = m['pca_first']
            # model_params = self.get_hparams(model_name, hparams)
            model_params = m['model_params']
            sampler_gamma = m['sampler_gamma']
            if box_type == 'none':
                n_estim = None
            # clf = get_model(model_name=model_name, model_params=model_params,
            #                 sampler_name=sampler_name, pca_bool=pca,
            #                 pca_first=pca_first, n_estim=n_estim,
            #                 box_type=box_type)
            clf = get_model(model_name=model_name, model_params=model_params,
                            sampler_name=sampler_name, pca_bool=pca,
                            pca_first=pca_first, n_estim=n_estim,
                            box_type=box_type, rbfsampler_gamma=sampler_gamma,
                            nystroem_gamma=sampler_gamma)

            n_splits_features = 30
            features = list(range(*features_range))
            if (features_range[1] - features_range[0]) > n_splits_features:
                features = np.linspace(*features_range,
                                       num=n_splits_features,
                                       dtype=np.int).tolist()

            if sampler_name == 'identity':
                features = None

            if sampler_name == 'identity':
                # train_score y test_score son floats
                train_score, test_score =\
                    get_non_sampling_model_scores(clf, dataset)

                lab = self.get_label(model_name, sampler_name, box_type,
                                     n_estim, pca, pca_first)
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
                train_score, test_score, errors =\
                    get_sampling_model_scores(clf, dataset, features)
                self.ERRORS.extend(errors)
                lab = self.get_label(model_name, sampler_name, box_type,
                                     n_estim, pca, pca_first)
                train_score['label'] = lab
                test_score['label'] = lab

            self.train_scores.append(train_score)
            self.test_scores.append(test_score)

        # return train_scores, test_scores

##############
# Deprecating
##############

#     def run_demo(self, dts_name, dts_size, features_range, models):
#         '''
#         Just reading from the arguments, returns a pair of list of dictionarys,
#         with the scores of the demo. Pair is (train, test)
#
#         Parameters
#         ----------
#         dts_name : str
#         dts_size : int
#         features_range : list of int
#             shape: [2], increasing order
#         models : list of dict
#             each dict is a model. Required keys: ['model_name', 'sampler',
#             'box_type', 'n_estim', 'pca']
#             Values of 'sampler' and 'box_type' are str or None
#
#         Returns
#         -------
#         (train_scores, test_scores) : tuple of list of dict
#             Dict with keys ['absi', 'ord', 'label']
#         '''
#         info_run = '''
# - Dataset: **{0}**
# - Size: **{1}**
#         '''
#         self.run_specific = info_run.format(dts_name, dts_size)
#         dataset = get_data(dts_name, n_ins=dts_size)
#         train_scores = []
#         test_scores = []
#
#         for m in models:
#             model_name = m['model_name']
#             sampler_name = m['sampler_name']
#             box_type = m['box_type']
#             n_estim = m['n_estim']
#             pca = m['pca']
#             pca_first = m['pca_first']
#             if box_type == 'none':
#                 n_estim = None
#             clf = get_model(model_name=model_name,
#                             sampler_name=sampler_name,
#                             pca_bool=pca,
#                             pca_first=pca_first,
#                             n_estim=n_estim,
#                             box_type=box_type)
#             n_splits_features = 30
#             features = list(range(*features_range))
#             if (features_range[1] - features_range[0]) > n_splits_features:
#                 features = np.linspace(*features_range,
#                                        num=n_splits_features,
#                                        dtype=np.int).tolist()
#
#             if sampler_name == 'identity':
#                 features = None
#
#             if sampler_name is 'identity':
#                 # train_score y test_score son floats
#                 train_score, test_score =\
#                     get_non_sampling_model_scores(clf, dataset)
#                 lab = self.get_label(model_name, sampler_name, box_type,
#                                      n_estim, pca, pca_first)
#                 train_score = {
#                     'absi': features_range,
#                     'ord': [train_score, train_score],
#                     'label': lab,
#                 }
#                 test_score = {
#                     'absi': features_range,
#                     'ord': [test_score, test_score],
#                     'label': lab,
#                 }
#             else:
#                 # train_score y test_score son diccionarios
#                 train_score, test_score, errors =\
#                     get_sampling_model_scores(clf, dataset, features)
#                 self.ERRORS.extend(errors)
#                 lab = self.get_label(model_name, sampler_name, box_type,
#                                      n_estim, pca, pca_first)
#                 train_score['label'] = lab
#                 test_score['label'] = lab
#
#             train_scores.append(train_score)
#             test_scores.append(test_score)
#
#         return train_scores, test_scores
