from demo_utils.generic_demo import Demo
# import ipywidgets as widgets
# from demo_utils.general import SUPPORTED_DATASETS
from demo_utils.general import get_data
# from demo_utils.learning import get_model_scores
from demo_utils.learning import get_model
from demo_utils.learning import get_non_sampling_model_scores
# from demo_utils.general import get_non_sampling_graph_from_scores


from ipywidgets import Button
# from ipywidgets import Dropdown
# from ipywidgets import RadioButtons
from ipywidgets import VBox

# TODO cambiar nombre dataset_selector por dts_selector


class Demo1(Demo):
    desc = """# Modelos básicos
    Permite ver cómo se comportan los modelos simples (Decision Tree, Logit,
    LinearSVM) con un dataset determinado. Permite saber si algunos problemas
    son más difíciles que otros para on modelo básico.
    """

    def __init__(self):
            self.run_bt = Button(description='Demo1', button_style='info',
                                 tooltip=self.desc)
            # self.dataset_selector = Dropdown(options=SUPPORTED_DATASETS,
            #                                  value=SUPPORTED_DATASETS[0],
            #                                  description='Dataset:')
            self.dataset_selector = self.get_default_dts_selector()
            # self.size_selector = RadioButtons(
            #     options=[1000, 2000, 5000, 10000],
            #     value=1000, description='Size')
            self.size_selector = self.get_default_size_selector()
            self.gui = VBox([self.size_selector, self.dataset_selector,
                             self.run_bt])
            super().__init__()

    def gui_to_data(self):
        '''
        Retorna un diccionario con todo lo que le hace falta para ejecutar la
        demos, y únicamente mira su gui
        Las claves siempre son str
        Todas las demos están obligadas a implementar esto

        Returns
        -------
        dict
            With keys: ['dts_name']
        '''
        d = {
            'dts_name': self.dataset_selector.value,
            'dts_size': self.size_selector.value,
        }
        return d

    def run_demo(self, dts_name, dts_size):
        '''
        Parameters
        ----------
        dts_name : str
            Name of the dataset to test. Must be one of SUPPORTED_DATASETS

        Returns
        -------
        (train_scores, test_scores) : tuple of list of dict
            Dicts have keys: ['absi', 'ord', 'label'], and 'absi' are valid
            numbers, -1 is not valid
        '''
        info_run = """
- Dataset: **{0}**
- Size: **{1}**
        """
        self.run_specific = info_run.format(dts_name, dts_size)
        # self.title = dts_name
        models_name = ['dt', 'logit', 'linear_svc']
        train_dicts = []
        test_dicts = []
        for model_name in models_name:
            dataset = get_data(dts_name, n_ins=dts_size)
            clf = get_model(model_name=model_name,
                            sampler_name='identity',
                            pca_bool=False,
                            box_type='none')
            train_score, test_score = get_non_sampling_model_scores(clf,
                                                                    dataset)
            train_score = {
                'absi': [0, 10],
                'ord': [train_score, train_score],
                'label': model_name,
            }
            test_score = {
                'absi': [0, 10],
                'ord': [test_score, test_score],
                'label': model_name,
            }
            train_dicts.append(train_score)
            test_dicts.append(test_score)
        return train_dicts, test_dicts
