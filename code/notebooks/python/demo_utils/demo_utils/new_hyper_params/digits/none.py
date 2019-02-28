def get_digits_none_dic():
    none_dic = {'dt': {'identity': {'min_impurity_decrease': 1e-07},
                       'rff': {'min_impurity_decrease': 1e-07, 'gamma': 1e-05},
                       'nystroem': {'min_impurity_decrease': 1e-07, 'gamma': 0.1}},
                'logit': {'identity': {'C': 1000},
                          'rff': {'C': 1000, 'gamma': 0.0001},
                          'nystroem': {'C': 1000, 'gamma': 0.0001}},
                'linear_svc': {'identity': {'C': 0.1},
                               'rff': {'C': 10, 'gamma': 0.001},
                               'nystroem': {'C': 100, 'gamma': 0.0001}}}
    return none_dic
