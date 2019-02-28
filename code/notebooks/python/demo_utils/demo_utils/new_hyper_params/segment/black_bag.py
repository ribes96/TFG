def get_segment_black_bag_dic():
    black_bag_dic = {'dt': {'identity': {'min_impurity_decrease': 0.001},
                            'rff': {'min_impurity_decrease': 0.001,
                                    'gamma': 1e-05},
                            'nystroem': {'min_impurity_decrease': 1e-05,
                                         'gamma': 0.001}},
                     'logit': {'identity': {'C': 1000},
                               'rff': {'C': 1000, 'gamma': 0.01},
                               'nystroem': {'C': 1000, 'gamma': 0.01}},
                     'linear_svc': {'identity': {'C': 1},
                                    'rff': {'C': 100, 'gamma': 0.01},
                                    'nystroem': {'C': 100, 'gamma': 0.1}}}
    return black_bag_dic
