def get_satellite_none_dic():
    none_dic = {'dt': {'identity': {'min_impurity_decrease': 1e-06},
                       'rff': {'min_impurity_decrease': 0.01, 'gamma': 0.001},
                       'nystroem': {'min_impurity_decrease': 0.01, 'gamma': 0.001}},
                'logit': {'identity': {'C': 1000},
                          'rff': {'C': 1000, 'gamma': 0.01},
                          'nystroem': {'C': 1000, 'gamma': 0.001}},
                'linear_svc': {'identity': {'C': 0.1},
                               'rff': {'C': 10, 'gamma': 0.01},
                               'nystroem': {'C': 1, 'gamma': 0.1}}}
    return none_dic
