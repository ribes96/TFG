import ipywidgets as widgets
from IPython.display import display
from IPython.display import Markdown as md
from demo_utils.general import get_non_sampling_error_graph_from_scores
from demo_utils.general import get_sampling_error_graph_from_scores

# from demo_utils.general import get_non_sampling_score_graph_from_scores
# from demo_utils.general import get_sampling_score_graph_from_scores

# TODO hacer funciones default_dataset_selector(), default_features_selector,
# etc. Que pase el genérico. Si alguno necesita una modificación particular,
# que se lo haga él.

# TODO si run_demo_with sampling se utiliza mucho (y creo que es el caso), no
# tiene sentido que cada subclase lo implemente por su cuenta. Quizá sale a
# cuenta hacerlo en el genérico, pero es algo demasiado específico


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
