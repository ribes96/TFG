def get_digits_none_sigest_dic():
    none_sigest_dic = {
        'dt': {'identity': {'min_impurity_decrease': 0.0001},
               'rff': {'min_impurity_decrease': 0.0001,
                       'gamma': 3.7129990238927874e-05},
               'nystroem': {'min_impurity_decrease': 0.001,
                            'gamma': 3.7129990238927874e-05}},
        'logit': {'identity': {'C': 1000},
                  'rff': {'C': 1000, 'gamma': 3.7129990238927874e-05},
                  'nystroem': {'C': 1000, 'gamma': 3.7129990238927874e-05}},
        'linear_svc': {'identity': {'C': 0.1},
                       'rff': {'C': 100, 'gamma': 3.7129990238927874e-05},
                       'nystroem': {'C': 100, 'gamma': 3.7129990238927874e-05}}
    }
    return none_sigest_dic
