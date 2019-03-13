def get_vowel_black_bag_dic():
    black_bag_dic = {'dt': {'identity': {'min_impurity_decrease': 0},
                            'rff': {'min_impurity_decrease': 0, 'gamma': 0.1},
                            'nystroem': {'min_impurity_decrease': 0,
                                         'gamma': 0.01}},
                     'logit': {'identity': {'C': 1000},
                               'rff': {'C': 1000, 'gamma': 0.1},
                               'nystroem': {'C': 1000, 'gamma': 0.1}},
                     'linear_svc': {'identity': {'C': 1000},
                                    'rff': {'C': 1000, 'gamma': 0.1},
                                    'nystroem': {'C': 1000, 'gamma': 0.1}}}
    return black_bag_dic
