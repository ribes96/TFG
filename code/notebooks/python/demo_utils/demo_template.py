from demo_utils.generic_demo import Demo
import ipywidgets as widgets


class Template(Demo):
    desc = 'description'
    run_bt = widgets.Button(description='Template', button_style='info')
    gui = run_bt

    def gui_to_data(self):
        '''
        Just reading from self.gui, return a dictionary with keys and values
        needed to run the demo. Keys are the arguments of run_demo
        '''
        pass

    def run_demo(self, arg1, arg2):
        '''
        Just reading from the arguments, return a pair of list of dictionarys,
        with the scores of the demo. Pair is (train, test)
        '''
        pass
