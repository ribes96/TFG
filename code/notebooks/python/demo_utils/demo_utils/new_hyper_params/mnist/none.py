def get_mnist_none_dic():
    none_dic = {'dt': {'identity': {'min_impurity_decrease': 0.0001},
                       'rff': {'min_impurity_decrease': 1e-08,
                               'gamma': 0.0001},
                       'nystroem': {'min_impurity_decrease': 0.001,
                                    'gamma': 1e-05}},
                'logit': {'identity': {'C': 1000},
                          'rff': {'C': 1000, 'gamma': 1e-05},
                          'nystroem': {'C': 1000, 'gamma': 0.0001}},
                'linear_svc': {'identity': {'C': 0.1},
                               'rff': {'C': 10, 'gamma': 0.0001},
                               'nystroem': {'C': 10, 'gamma': 0.0001}}}

    return none_dic
