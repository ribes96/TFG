def get_pen_digits_grey_ens_dic():
    grey_ens_dic = {'dt': {'identity': {'min_impurity_decrease': 0},
                           'rff': {'min_impurity_decrease': 0, 'gamma': 0.001},
                           'nystroem': {'min_impurity_decrease': 0,
                                        'gamma': 0.001}},
                    'logit': {'identity': {'C': 1000},
                              'rff': {'C': 1000, 'gamma': 0.1},
                              'nystroem': {'C': 1000, 'gamma': 0.1}},
                    'linear_svc': {'identity': {'C': 1000},
                                   'rff': {'C': 1000, 'gamma': 0.01},
                                   'nystroem': {'C': 1000, 'gamma': 0.1}}}
    return grey_ens_dic
