def get_covertype_none_dic():
    none_dic = {'dt': {'identity': {'min_impurity_decrease': 0.01},
                       'rff': {'min_impurity_decrease': 1e-07, 'gamma': 0.001},
                       'nystroem': {'min_impurity_decrease': 1e-07, 'gamma': 0.001}},
                'logit': {'identity': {'C': 1000},
                          'rff': {'C': 1000, 'gamma': 0.01},
                          'nystroem': {'C': 1000, 'gamma': 0.001}},
                'linear_svc': {'identity': {'C': 10},
                               'rff': {'C': 100, 'gamma': 0.01},
                               'nystroem': {'C': 100, 'gamma': 0.01}}}
    return none_dic
