def get_pen_digits_none_dic():
    none_dic = {'dt': {'identity': {'min_impurity_decrease': 1e-07},
                       'rff': {'min_impurity_decrease': 1e-07, 'gamma': 0.001},
                       'nystroem': {'min_impurity_decrease': 1e-05, 'gamma': 0.0001}},
                'logit': {'identity': {'C': 1000},
                          'rff': {'C': 1000, 'gamma': 0.01},
                          'nystroem': {'C': 1000, 'gamma': 0.1}},
                'linear_svc': {'identity': {'C': 1},
                               'rff': {'C': 100, 'gamma': 0.01},
                               'nystroem': {'C': 100, 'gamma': 0.01}}}
    return none_dic
