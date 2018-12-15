from demo_utils.generic_demo import Demo

from ipywidgets import Button
# from ipywidgets import Dropdown
# from ipywidgets import RadioButtons
# from ipywidgets import IntRangeSlider
# from ipywidgets import VBox
# from ipywidgets import HBox
# from ipywidgets import Label
# from ipywidgets import Layout
# from ipywidgets import IntSlider
# from ipywidgets import Checkbox


class Template(Demo):
    desc = '# Description of the Template'

    def __init__(self):
        self.run_bt = Button(description='Template', button_style='info')
        self.gui = self.run_bt
        super().__init__()

    def gui_to_data(self):
        '''
        Just reading from self.gui, return a dictionary with keys and values
        needed to run the demo. Keys are the arguments of run_demo

        Es importante que genere información, pero no estructuras abstractas
        que son resultado de llamar a algún método. Que contenga tipos de
        vanila python como numeros, strings, listas y diccionarios
        '''
        ret_dict = {
            'arg1': 'value1',
            'arg2': 'value2'
        }
        return ret_dict
        pass

    def run_demo(self, arg1, arg2):
        '''
        Just reading from the arguments, return a pair of list of dictionarys,
        with the scores of the demo. Pair is (train, test)
        '''
        info_run = '''
- Arg 1: **{0}**
- Arg 2: **{1}**
        '''
        self.run_specific = info_run.format(arg1, arg2)
        score = {
            'absi': [0, 10],
            'ord': [0.4, 0.4],
            'label': 'template label',
        }
        return [score], [score]
