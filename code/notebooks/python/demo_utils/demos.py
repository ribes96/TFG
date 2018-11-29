# This file is ready to be removed. In here just in case...


# # from demo_utils.learning import get_all_model_scores
# from demo_utils.learning import get_model_scores
# # from demo_utils.user_interface import get_gui
#
# from demo_utils.general import SUPPORTED_DATASETS
# from demo_utils.general import get_data
#
# from demo_utils.demos_resources import demos_resources
#
# from demo_utils.learning import get_graph_from_scores
#
# from IPython.display import display
#
# import ipywidgets as widgets
#
#
# def demo1(interactive=True, dts_name=None):
#     '''
#     Los modelos normales, sin ningún cambio
#
#     Parameters
#     -----------
#     inteactive: bool, si es True, sacará un botón y los widgets necesarios,
#     y la demo se ejecutará con el botón; si es False, sacará los resultados
#     directamente
#
#     dts_name: str or None, el nombre del dataset en el que ejecutar la demo. Si
#     es None, sacará un Dropdown para elegirlo
#
#     Generará un error interactive = False y no indicar el dataset
#
#     Returns
#     -------
#     Si inteactive = True, un HBox para ejecutar la demo. Si False, muestra la
#     gráfica y retorna None
#     '''
#
#     if interactive is False and dts_name is None:
#         raise ValueError("dts_name was not given")
#
#     DEMO_NUMBER = 1
#     DEMO_SPEC = demos_resources[DEMO_NUMBER]
#     DEMO_NAME = DEMO_SPEC['name']
#     DEMO_DESC = DEMO_SPEC['desc']
#     # una instancia de los datos que necesita para ejecutar uno de los
#     # modelos de la demo. Estará en demos resources junto con svm y logit
#     DEMO_MODELS = demos_resources[DEMO_NUMBER]['models']
#
#     def button_action(e=None):
#         # name = dts_name if not interactive else dts_selector.value
#         name = dts_selector.value if interactive else dts_name
#         # train_scores, test_scores = run_demo(dts_selector.value)
#         train_scores, test_scores = run_demo(name)
#
#         gui_data = {
#             # 'dataset_selector': dts_selector.value,
#             'dataset_selector': name,
#             'size_selector': DEMO_SPEC['n_ins'],
#         }
#
#         # fig = get_graph_from_scores(train_scores, test_scores, gui_data)
#         fig = get_graph_from_scores(train_scores, test_scores)
#         with out:
#             display(fig)
#
#     def run_demo(dts_name):
#         '''
#         Returns a pair of lists of dictionarys, first train, then test
#
#         Parameters
#         ----------
#         dts_name : str
#             name of dataset, one of SUPPORTED_DATASETS = ["covertype",
#             "digits", "fall_detection", ("mnist"), "pen_digits", "satellite",
#             "segment", "vowel"]
#
#         Returns
#         -------
#         (train_socres, test_scores) : tuple of list
#             The lists contain dictionarys with keys ['absi', 'ord', 'label']
#         '''
#         train_scores = []
#         test_scores = []
#         data = get_data(dts_name, n_ins=DEMO_SPEC['n_ins'])
#         # data = get_data(dts_name, n_ins=1000)
#         for model in DEMO_MODELS:
#             model['dataset'] = data
#             train_dic, test_dic = get_model_scores(**model)
#             train_scores.append(train_dic)
#             test_scores.append(test_dic)
#         return train_scores, test_scores
#
#     bt = widgets.Button(
#         description=DEMO_NAME,
#         button_style='info',
#         tooltip=DEMO_DESC,
#     )
#
#     dts_selector = widgets.Dropdown(
#         options=SUPPORTED_DATASETS,
#         value=SUPPORTED_DATASETS[0],
#         description='Dataset:',
#     )
#     bt.on_click(button_action)
#
#     gui = widgets.HBox([dts_selector, bt])
#     out = widgets.Output()
#     if interactive:
#         with out:
#             display(DEMO_DESC)
#         return widgets.VBox([gui, out])
#     else:
#         button_action()
#         return out
