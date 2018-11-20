import ipywidgets as widgets


def get_new_model_bar():
    '''
    Returns
    -------
    Returns a new HBox with the widgets to define a new training model
    '''
    model_selector = widgets.Dropdown(
        options=['dt', 'logit', 'linear_svc'],
        value='dt',
        layout = widgets.Layout(flex = '0 3 auto'),
        #description=':',
    )

    sampler_selector = widgets.Dropdown(
        options=['None', 'rbf', 'nystroem'],
        value='None',
        layout = widgets.Layout(flex = '1 3 auto'),
        #description=':',
    )
    box_type_selector = widgets.Dropdown(
        options=['None', 'black', 'grey'],
        value='None',
        layout = widgets.Layout(flex = '1 3 auto'),
        #description=':',
    )
    '''
    features_selector = widgets.IntRangeSlider(
        value=[30, 100],
        min=30,
        max=400,
        step=10,
        layout = Layout(flex = '0 1 auto'),
        #description=':',
    )
    '''
    n_estimators_selector = widgets.IntSlider(
        value=30,
        min=2,
        max=200,
        step=1,
        layout = widgets.Layout(flex = '1 1 auto'),
        #description=':',
    )
    pca_checkbox = widgets.Checkbox(
        value=False,
        layout = widgets.Layout(flex = '0 3 auto'),
        #description='',
    )
    hb = widgets.HBox([
        model_selector,
        sampler_selector,
        box_type_selector,
        #features_selector,
        n_estimators_selector,
        pca_checkbox,
    ])
    return hb

def get_gui():
    '''
    Return a VBox which is the user interface

    Probably, return also a dictionary with elements in the gui
    '''
    dataset_selector = widgets.Dropdown(
        options = [
            "covertype",
            "digits",
            "fall_detection",
            #"mnist",
            # De momento he tenido que quitarlo, por demasiado pesado
            # Todo reducir el dataset y añadirlo de nuevo
            "pen_digits",
            "satellite",
            "segment",
            "vowel",
            ],
        value = 'digits',
        descripttion = 'Dataset:'
    )




    size_selector = widgets.RadioButtons(
        options=[1000,2000,5000,10000],
        value=2000,
        #description='Pizza topping:',
        disabled=False,
        #orientation = 'vertical'
    )
    cool_size_selector = widgets.VBox([widgets.Label("Size of the dataset"), size_selector])

    def add_model_bar(m):
        '''
        Append a new model bar to m, which has the same values as the last
        model bar in m
        Parameters
        ----------
        m: Is a VBox containing 1 or more HBox describing the new model
        '''
        if len(m.children) < 1:
            raise ValueError('At least one model bar is needed')
        copy_bar = m.children[-1]

        new_model_bar = get_new_model_bar()

        for i,c in enumerate(copy_bar.children):
            new_model_bar.children[i].value = c.value

        m.children = tuple(list(m.children) + [new_model_bar])

    def add_model_bar_wraper(e):
        add_model_bar(models_bar)

    add_model_bar_bt = widgets.Button(
        description='Add model',
        button_style='success', # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Add a new moder bar to train',
        #icon='check'
    )
    add_model_bar_bt.on_click(add_model_bar_wraper)

    def remove_model_bar(m):
        '''
        Remove the las model bar of m, if there are at least 2
        Parameters
        ----------
        m: Is a VBox containing 2 or more HBox describing models
        '''
        if len(m.children) < 2:
            raise ValueError('minimum number of model bars reached')
        m.children = tuple(list(m.children)[:-1])



    def remove_model_bar_wraper(e):
        if len(models_bar.children) > 1:
            remove_model_bar(models_bar)

    remove_model_bar_bt = widgets.Button(
        description='Remove model',
        button_style='warning', # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Remove the las model bar, if possible',
        #icon='check'
    )
    remove_model_bar_bt.on_click(remove_model_bar_wraper)


    features_selector = widgets.IntRangeSlider(
        value=[30, 100],
        min=30,
        max=400,
        step=10,
        layout = widgets.Layout(flex = '0 1 auto'),
        #description=':',
    )

    models_bar = widgets.VBox([get_new_model_bar()])

    headers = widgets.HBox([
        widgets.Label("Model"),
        widgets.Label("Sampling"),
        widgets.Label("Box Type"),
        widgets.Label("Number estimators"),
        widgets.Label("PCA"),
    ], layout = widgets.Layout(justify_content = 'space-between'))

    cool_models_bar = widgets.VBox([headers, models_bar])


    calculate_bt = widgets.Button(
        description='Calculate',
        button_style='success', # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Calculate the models',
        #icon='check'
    )
    # calculate_bt.on_click(calculate_bt_wrapper)

    clear_output_button = widgets.ToggleButton(
        value=True,
        description='Clear Previous',
        button_style='', # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Clear Previous Output',
        icon='check'
    )

    progress_bar = widgets.IntProgress(
        value=0,
        min=0,
        max=10,
        step=1,
        description='Calculating:',
        bar_style='info', # 'success', 'info', 'warning', 'danger' or ''
        orientation='horizontal'
    )

    sub_progress_bar = widgets.IntProgress(
        value=0,
        min=0,
        max=10,
        step=1,
        #description='',
        bar_style='info', # 'success', 'info', 'warning', 'danger' or ''
        orientation='horizontal'
    )


    gui = widgets.VBox([
        dataset_selector,
        cool_size_selector,
        widgets.HBox([add_model_bar_bt, remove_model_bar_bt]),
        features_selector,
        cool_models_bar,
        calculate_bt,
        clear_output_button,
        progress_bar,
        sub_progress_bar,
    ])

    gui_dic = {
        'calculate_bt': calculate_bt,
        'models_bar': models_bar,
        'dataset_selector': dataset_selector,
        'size_selector': size_selector,
        'features_selector': features_selector,
        'clear_output_button': clear_output_button,
        'progress_bar': progress_bar,
        'sub_progress_bar': sub_progress_bar,
    }

    return gui, gui_dic
