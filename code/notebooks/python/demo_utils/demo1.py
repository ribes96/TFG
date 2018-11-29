from demo_utils.generic_demo import Demo
import ipywidgets as widgets
from demo_utils.general import SUPPORTED_DATASETS
from demo_utils.general import get_data
# from demo_utils.learning import get_model_scores
from demo_utils.learning import get_model
from demo_utils.learning import get_non_sampling_model_scores

# todo sobreescribir el button_action del padre para que haga un bar-plot
#   en vez de el genérico
# ahora mismo la interfaz es súper fea, hacerla más bonita


class Demo1(Demo):
    desc = """
    Permite ver cómo se comportan los modelos simples (Decision Tree, Logit,
    LinearSVM) con un dataset determinado. De esta manera sabremos si un
    problema es más fácil para un modelo o para otro.
    """

    run_bt = widgets.Button(description='Demo1', button_style='info',
                            tooltip=desc)
    dataset_selector = widgets.Dropdown(options=SUPPORTED_DATASETS,
                                        value=SUPPORTED_DATASETS[0],
                                        description='Dataset:')
    size_selector = widgets.RadioButtons(options=[1000, 2000, 5000, 10000],
                                         value=1000)
    gui = widgets.HBox([dataset_selector, size_selector, run_bt])

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
