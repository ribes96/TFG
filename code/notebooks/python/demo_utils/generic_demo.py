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
# from ipywidgets import VBox
# from ipywidgets import HBox
# from ipywidgets import Label
# from ipywidgets import Layout
from ipywidgets import IntSlider
from ipywidgets import Checkbox

# from demo_utils.general import get_non_sampling_score_graph_from_scores
# from demo_utils.general import get_sampling_score_graph_from_scores

# TODO hacer funciones default_dataset_selector(), default_features_selector,
# etc. Que pase el genérico. Si alguno necesita una modificación particular,
# que se lo haga él.

# TODO si run_demo_with sampling se utiliza mucho (y creo que es el caso), no
# tiene sentido que cada subclase lo implemente por su cuenta. Quizá sale a
# cuenta hacerlo en el genérico, pero es algo demasiado específico

# TODO que tolas las demos utilicen el método default para sacar algo de la
# interfaz, en vez de crearlo ellas. Si acaso que luego le pongan un layout
# No olvidar las de los model de la demo0/6

# TODO probablemente la mayoría de demos van mal porque hemos cambiado los
# box_types


class Demo:
    '''
    A generic Class to create Demos. Other Demos can use this as a super class
    and at minimum they just need to implement gui_to_data and run_demo
    '''

    def __init__(self):
        self.graph_output = widgets.Output()
        self.run_bt.on_click(self.button_action)
        # self.title = 'Titulo genérico'
        self.run_specific = 'Info about a run'

    def _repr_html_(self):
        display(md(self.desc))
        display(self.gui)
        display(self.graph_output)
        return ''

    def run_demo(self):
        print("run_demo must be implemented by the child demo")

    def non_interactive(self, **argw):
        display(md(self.desc))
        train_scores, test_scores = self.run_demo(**argw)
        fig = self.get_generic_graph_from_scores(train_scores, test_scores)
        # fig.suptitle(self.title)
        display(md(self.run_specific))
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

    def button_action(self, e=None):
        '''
        Generic function to perform when the Demo button is pressed. It must
        get data from the widgets, run the demo and display a graph. Be sure to
        do all of that if you want a child to override this function
        '''
        data = self.gui_to_data()
        train_scores, test_scores = self.run_demo(**data)

        fig = self.get_generic_graph_from_scores(train_scores, test_scores)

        # fig.suptitle(self.title)
        self.graph_output.clear_output(wait=True)
        with self.graph_output:
            display(md(self.run_specific))
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
        features_selector = IntRangeSlider(value=[30, 100], min=30, max=2000,
                                           step=1)
        return features_selector

    def get_default_sampler_selector(self):
        # TODO las descripciones se las pone cada uno
        # sampler_selector = Dropdown(options=['None', 'rbf', 'nystroem'],
        #                             value='None', description='Sampler')
        sampler_selector = Dropdown(options=['None', 'rbf', 'nystroem'],
                                    value='None')
        return sampler_selector

    def get_default_box_type_selector(self):
        # más adelante no será necesario
        # DD_lay = Layout(width='90px', margin='8px 2px 8px 2px')
        # box_type_selector = Dropdown(
        #     options=['black_bag', 'grey_bag', 'black_ens', 'grey_ens', 'None'],
        #     value='None', layout=DD_lay)
        box_type_selector = Dropdown(
            options=['black_bag', 'grey_bag', 'black_ens', 'grey_ens', 'None'],
            value='None')
        return box_type_selector

    def get_default_n_estimators_selector(self):
        # TODO el layout habrá que quitarlo, y el tema del visibility también
        # n_estimators_selector = IntSlider(value=30, min=2, max=200, step=1,
        #                                   disabled=True,
        #                                   layout=Layout(
        #                                         width='300px',
        #                                         visibility='hidden',
        #                                         margin='8px 2px 8px 2px'))
        n_estimators_selector = IntSlider(value=30, min=2, max=200, step=1)
        return n_estimators_selector

    def get_default_pca_checkbox(self):
        pca_checkbox = Checkbox(value=False, indent=False)
        return pca_checkbox

    def get_default_pca_order_selector(self):
        pca_order_selector = RadioButtons(
            options={'PCA First': True, 'Sampler First': False}, value=False)
        return pca_order_selector
