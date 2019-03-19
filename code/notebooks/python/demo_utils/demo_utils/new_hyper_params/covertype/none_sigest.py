def get_covertype_none_sigest_dic():
    none_sigest_dic = {
        'dt': {'identity': {'min_impurity_decrease': 0.001},
               'rff': {'min_impurity_decrease': 0.0001,
                       'gamma': 0.0007733047113058188},
               'nystroem': {'min_impurity_decrease': 0.001,
                            'gamma': 0.0007733047113058188}},
        'logit': {'identity': {'C': 1000},
                  'rff': {'C': 1000, 'gamma': 0.0007733047113058188},
                  'nystroem': {'C': 1000, 'gamma': 0.0007733047113058188}},
        'linear_svc': {'identity': {'C': 1},
                       'rff': {'C': 100, 'gamma': 0.0007733047113058188},
                       'nystroem': {'C': 100, 'gamma': 0.0007733047113058188}}}
    return none_sigest_dic
