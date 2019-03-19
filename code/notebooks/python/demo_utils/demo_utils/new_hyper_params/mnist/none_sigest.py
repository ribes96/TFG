def get_mnist_none_sigest_dic():
    none_sigest_dic = {
        'dt': {'identity': {'min_impurity_decrease': 0.001},
               'rff': {'min_impurity_decrease': 0.001,
                       'gamma': 2.777912847345097e-07},
               'nystroem': {'min_impurity_decrease': 0.0001,
                            'gamma': 2.777912847345097e-07}},
        'logit': {'identity': {'C': 1000},
                  'rff': {'C': 1000, 'gamma': 2.777912847345097e-07},
                  'nystroem': {'C': 1000, 'gamma': 2.777912847345097e-07}},
        'linear_svc': {'identity': {'C': 0.1},
                       'rff': {'C': 100, 'gamma': 2.777912847345097e-07},
                       'nystroem': {'C': 100, 'gamma': 2.777912847345097e-07}}}
    return none_sigest_dic
