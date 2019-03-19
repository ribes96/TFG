def get_vowel_none_sigest_dic():
    none_sigest_dic = {
        'dt': {'identity': {'min_impurity_decrease': 0.001},
               'rff': {'min_impurity_decrease': 0.001,
                       'gamma': 0.0011455688869636692},
               'nystroem': {'min_impurity_decrease': 0.001,
                            'gamma': 0.0011455688869636692}},
        'logit': {'identity': {'C': 1000},
                  'rff': {'C': 1000, 'gamma': 0.0011455688869636692},
                  'nystroem': {'C': 1000, 'gamma': 0.0011455688869636692}},
        'linear_svc': {'identity': {'C': 10},
                       'rff': {'C': 100, 'gamma': 0.0011455688869636692},
                       'nystroem': {'C': 100, 'gamma': 0.0011455688869636692}}}
    return none_sigest_dic
