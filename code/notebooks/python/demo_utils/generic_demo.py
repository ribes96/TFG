import ipywidgets as widgets
from IPython.display import display
# from demo_utils.general import SUPPORTED_DATASETS
from demo_utils.learning import get_graph_from_scores
from IPython.display import Markdown as md
# from demo_utils.general import get_data
# from demo_utils.learning import get_model_scores
# from demo_utils.demos_resources import demos_resources
# from demo_utils.user_interface import get_new_model_bar
# from demo_utils.learning import get_model
# from demo_utils.learning import get_non_sampling_model_scores
# from demo_utils.learning import get_sampling_model_scores

# todo añadir mnist reducido
# todo hacer el modo no interactivo


class Demo:
    '''
    A generic Class to create Demos. Other Demos can use this as a super class
    and at minimum they just need to implement gui_to_data and run_demo
    '''

    def __init__(self):
        self.graph_output = widgets.Output()
        self.run_bt.on_click(self.button_action)

    def _repr_html_(self):
        display(md(self.desc))
        display(self.gui)
        display(self.graph_output)
        return ''

    def run_demo(self):
        print("run_demo must be implemented by the child demo")

    def button_action(self, e=None):
        '''
        Generic function to perform when the Demo button is pressed. It must
        get data from the widgets, run the demo and display a graph. Be sure to
        do all of that if you want a child to override this function
        '''
        data = self.gui_to_data()
        train_scores, test_scores = self.run_demo(**data)
        fig = get_graph_from_scores(train_scores, test_scores)
        # todo ahora se podría poner un título

        self.graph_output.clear_output(wait=True)
        with self.graph_output:
            display(fig)