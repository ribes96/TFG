import ipywidgets as widgets
from IPython.display import display
from IPython.display import Markdown as md
from demo_utils.general import get_non_sampling_error_graph_from_scores
from demo_utils.general import get_sampling_error_graph_from_scores
from demo_utils.general import SUPPORTED_DATASETS

# from ipywidgets import Button
from ipywidgets import Dropdown
from ipywidgets import RadioButtons
from ipywidgets import IntRangeSlider
from ipywidgets import FloatLogSlider
from ipywidgets import FloatSlider
from ipywidgets import VBox
from ipywidgets import HBox
# from ipywidgets import Label
from ipywidgets import Layout
from ipywidgets import IntSlider
from ipywidgets import Checkbox
from ipywidgets import HTML
from ipywidgets import Accordion

import math

# from demo_utils.general import get_non_sampling_score_graph_from_scores
# from demo_utils.general import get_sampling_score_graph_from_scores

# TODO si run_demo_with sampling se utiliza mucho (y creo que es el caso), no
# tiene sentido que cada subclase lo implemente por su cuenta. Quizá sale a
# cuenta hacerlo en el genérico, pero es algo demasiado específico

# TODO que tolas las demos utilicen el método default para sacar algo de la
# interfaz, en vez de crearlo ellas. Si acaso que luego le pongan un layout
# No olvidar las de los model de la demo0/6

# TODO probablemente la mayoría de demos van mal porque hemos cambiado los
# box_types

# TODO cambiar nombres de button_action2

# TODO rbf es un mal nombre para el sampler RFF, colisiona con el nombre del
# kernel directamente. Sustituirlo por fourier


class Demo:
    '''
    A generic Class to create Demos. Other Demos can use this as a super class
    and at minimum they just need to implement gui_to_data and run_demo
    '''

    def __init__(self):
        self.graph_output = widgets.Output()
        # self.run_bt.on_click(self.button_action)
        self.run_bt.on_click(self.button_action2)
        # self.title = 'Titulo genérico'
        self.run_specific = 'Info about a run'

        # En caso de que falle algo, pondrán aquí dentro un diccionario con la
        # información del fallo, para que luego se pueda hacer un poco de
        # forensics
        self.ERRORS = []

        self.train_scores = []
        self.test_scores = []

    def _repr_html_(self):
        display(md(self.desc))
        display(self.gui)
        display(self.graph_output)
        return ''
        # return self.gui

    def get_run_specific_widget(self, d):
        '''
        Returns a widget (accordion) full of labels showing the info in d, with
        the key in bold
        '''
        labels =\
            [HTML(value='<strong>{}</strong>: {}'.format(k, d[k])) for k in d]
        # v1 = VBox(labels)
        lay = Layout(margin='0px 20px 0px 0px')
        # margen a la derecha
        n_parts = 4
        p_len = math.ceil(len(labels) / n_parts)
        parts = [labels[i:i+p_len] for i in range(0, len(labels), p_len)]
        vboxes = [VBox(i, layout=lay) for i in parts]
        h = HBox(vboxes)

        # ac = Accordion([v1])
        ac = Accordion([h])
        ac.set_title(0, 'Info run')
        return ac

    def run_demo(self):
        print("run_demo must be implemented by the child demo")

    # def non_interactive(self, argw):
    #     print(type(argw))
    #     display(md(self.desc))
    #     train_scores, test_scores = self.run_demo2(**argw)
    #     fig = self.get_generic_graph_from_scores(train_scores, test_scores)
    #     # fig.suptitle(self.title)
    #     display(md(self.run_specific))
    #     display(fig)

    def non_interactive(self, **argw):
        display(md(self.desc))
        self.run_demo(**argw)
        fig = self.get_generic_graph_from_scores(self.train_scores,
                                                 self.test_scores)
        # fig.suptitle(self.title)
        # display(md(self.run_specific))
        display(self.run_specific)
        display(fig)

    def all_is_non_sampling(self, train_scores, test_scores):
        '''
        Returns if every score in train_scores and test_scores is non-sampling,
        based on 'absi' having only two elements
        '''
        is_all = True
        for tr_m, te_m in zip(train_scores, test_scores):
            if len(tr_m['absi']) > 2 or len(te_m['absi']) > 2:
                is_all = False
                break
        return is_all

    def get_generic_graph_from_scores(self, train_scores, test_scores):
        '''
        This is just a wrapper to call get_graph_from_scores or
        get_graph_non_sampling_from_scores depending on the type of graph is
        desired. By default it calls get_graph_from_scores, but any child class
        can overwrite this mehod
        '''
        # Está puesto para que de el error_graph, pero se puede dar el
        # score_graph
        if self.all_is_non_sampling(train_scores, test_scores):
            # return get_non_sampling_graph_from_scores(train_scores,
            #                                           test_scores)
            return get_non_sampling_error_graph_from_scores(train_scores,
                                                            test_scores)
        else:
            # return get_sampling_graph_from_scores(train_scores, test_scores)
            return get_sampling_error_graph_from_scores(train_scores,
                                                        test_scores)

    # deprecating
    # def button_action(self, e=None):
    #     '''
    #     Generic function to perform when the Demo button is pressed. It must
    #     get data from the widgets, run the demo and display a graph. Be sure
    # to
    #     do all of that if you want a child to override this function
    #     '''
    #     data = self.gui_to_data()
    #     train_scores, test_scores = self.run_demo(**data)
    #
    #     fig = self.get_generic_graph_from_scores(train_scores, test_scores)
    #
    #     # fig.suptitle(self.title)
    #     self.graph_output.clear_output(wait=True)
    #     with self.graph_output:
    #         display(md(self.run_specific))
    #         display(fig)

    # pendiente cambiar el nombre, quitando el 2
    def button_action2(self, e=None):
        '''
        Generic function to perform when the Demo button is pressed. It must
        get data from the widgets, run the demo and display a graph. Be sure to
        do all of that if you want a child to override this function
        '''
        # TODO quizá data y fig también pueden pertenecer a la clase
        data = self.gui_to_data()
        self.run_demo(**data)

        fig = self.get_generic_graph_from_scores(self.train_scores,
                                                 self.test_scores)

        self.graph_output.clear_output(wait=True)
        with self.graph_output:
            # display(md(self.run_specific))
            display(self.run_specific)
            display(fig)

    def get_default_dts_selector(self):
        # dts_selector = Dropdown(options=SUPPORTED_DATASETS,
        #                         value=SUPPORTED_DATASETS[0],
        #                         description='Dataset:')
        dts_selector = Dropdown(options=SUPPORTED_DATASETS,
                                value=SUPPORTED_DATASETS[0])
        return dts_selector

    def get_default_size_selector(self):
        size_selector = RadioButtons(options=[1000, 2000, 5000, 10000],
                                     value=1000, description='Size')
        return size_selector

    def get_default_model_selector(self):
        # DD_lay = Layout(width='90px', margin='8px 2px 8px 2px')
        # model_selector = Dropdown(options=['dt', 'logit', 'linear_svc'],
        #                           value='dt', layout=DD_lay)
        model_selector = Dropdown(options=['dt', 'logit', 'linear_svc'],
                                  value='dt')
        return model_selector

    def get_default_features_selector(self):
        # features_selector = IntRangeSlider(value=[30, 100], min=30, max=2000,
        #                                    step=1,
        #                                    layout=Layout(width='950px'))
        features_selector = IntRangeSlider(value=[30, 100], min=30, max=1000,
                                           step=10)
        return features_selector

    def get_default_sampler_selector(self):
        sampler_selector = Dropdown(options=['None', 'rbf', 'nystroem'],
                                    value='None')
        return sampler_selector

    def get_default_box_type_selector(self):
        box_type_selector = Dropdown(
            options=['black_bag', 'grey_bag', 'black_ens', 'grey_ens', 'None'],
            value='None')
        return box_type_selector

    def get_default_n_estimators_selector(self):
        n_estimators_selector = IntSlider(value=30, min=2, max=200, step=1)
        return n_estimators_selector

    def get_default_pca_checkbox(self):
        pca_checkbox = Checkbox(value=False, indent=False)
        return pca_checkbox

    def get_default_pca_order_selector(self):
        pca_order_selector = RadioButtons(
            options={'PCA First': True, 'Sampler First': False}, value=False)
        return pca_order_selector

    def get_dt_hp_tab(self):
        '''
        Returns the tab to tune hyper-parameters of DecisionTree
        '''
        self.dt_max_depth_selector = FloatLogSlider(value=100, base=10, min=0,
                                                    max=4, step=1,
                                                    description='Max depth')
        self.dt_min_samples_split_selector =\
            IntSlider(value=3, min=1, max=10, step=1,
                      description='Min Samples Split:')
        self.dt_min_samples_leaf_selector =\
            IntSlider(value=1, min=1, max=10, step=1,
                      description='Min Samples Leaf:')
        self.dt_min_weight_fraction_leaf_selector =\
            FloatSlider(value=.0, min=0, max=.5, step=.1,
                        description='min_weight_fraction_leaf:')
        self.dt_max_leaf_nodes_selector =\
            FloatLogSlider(value=1000, base=10, min=0, max=4, step=1,
                           description='Max leaf nodes')
        self.dt_min_impurity_decrease_selector =\
            FloatSlider(value=.0, min=0, max=1, step=.1,
                        description='min_impurity_decrease:')
        tab = VBox([
            self.dt_max_depth_selector,
            self.dt_min_samples_split_selector,
            self.dt_min_samples_leaf_selector,
            self.dt_min_weight_fraction_leaf_selector,
            self.dt_max_leaf_nodes_selector,
            self.dt_min_impurity_decrease_selector,
            ])
        return tab

    def get_logit_hp_tab(self):
        '''
        Returns the tab to tune h-params. of LogisticRegression
        '''
        self.logit_C_selector = FloatLogSlider(value=1000, base=10, min=-5,
                                               max=5, step=1, description='C')
        tab = VBox([self.logit_C_selector])
        return tab

    def get_linearsvc_hp_tab(self):
        '''
        Returns the tab to tune h-params. of LinearSVC
        '''
        self.linearsvc_C_selector = IntSlider(value=5, min=1, max=10, step=1,
                                              description='C')
        tab = VBox([self.linearsvc_C_selector])
        return tab

    def get_samplers_params_tab(self):
        # self.rbf_gamma_selector = FloatSlider(value=1, min=0.1, max=100,
        #                                       step=0.1,
        #                                       description='RBFSampler gamma')
        # self.nystroem_gamma_selector =\
        #     FloatSlider(value=1, min=0.1, max=100, step=0.1,
        #                 description='Nystroem gamma')

        self.rbf_gamma_selector = FloatLogSlider(value=1, base=10, min=-5, max=3,
                                              step=1,
                                              description='RBFSampler gamma')
        self.nystroem_gamma_selector =\
            FloatLogSlider(value=1, base=10, min=-5, max=3, step=1,
                        description='Nystroem gamma')
        tab = VBox([self.rbf_gamma_selector, self.nystroem_gamma_selector])
        return tab
