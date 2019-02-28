def get_vowel_none_dic():
    none_dic = {'dt': {'identity': {'min_impurity_decrease': 1e-08},
                       'rff': {'min_impurity_decrease': 1e-07, 'gamma': 0.001},
                       'nystroem': {'min_impurity_decrease': 1e-08, 'gamma': 0.01}},
                'logit': {'identity': {'C': 1000},
                          'rff': {'C': 1000, 'gamma': 0.1},
                          'nystroem': {'C': 1000, 'gamma': 0.1}},
                'linear_svc': {'identity': {'C': 100},
                               'rff': {'C': 10, 'gamma': 0.1},
                               'nystroem': {'C': 100, 'gamma': 0.1}}}
    return none_dic
