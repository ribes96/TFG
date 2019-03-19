def get_satellite_none_sigest_dic():
    none_sigest_dic = {
        'dt': {'identity': {'min_impurity_decrease': 0.001},
               'rff': {'min_impurity_decrease': 0.001,
                       'gamma': 6.199229262072869e-05},
               'nystroem': {'min_impurity_decrease': 0.01,
                            'gamma': 6.199229262072869e-05}},
        'logit': {'identity': {'C': 1000},
                  'rff': {'C': 1000, 'gamma': 6.199229262072869e-05},
                  'nystroem': {'C': 1000, 'gamma': 6.199229262072869e-05}},
        'linear_svc': {'identity': {'C': 10},
                       'rff': {'C': 100, 'gamma': 6.199229262072869e-05},
                       'nystroem': {'C': 100, 'gamma': 6.199229262072869e-05}}}
    return none_sigest_dic
