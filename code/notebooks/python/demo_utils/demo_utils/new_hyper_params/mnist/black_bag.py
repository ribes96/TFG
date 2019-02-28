def get_mnist_black_bag_dic():
    black_bag_dic = {'dt': {'identity': {'min_impurity_decrease': 1e-05},
                            'rff': {'min_impurity_decrease': 1e-08,
                                    'gamma': 0.0001},
                            'nystroem': {'min_impurity_decrease': 0.0001,
                                         'gamma': 1e-05}},
                     'logit': {'identity': {'C': 1000},
                               'rff': {'C': 1000, 'gamma': 1e-05},
                               'nystroem': {'C': 1000, 'gamma': 1e-05}},
                     'linear_svc': {'identity': {'C': 0.1},
                                    'rff': {'C': 10, 'gamma': 0.0001},
                                    'nystroem': {'C': 10, 'gamma': 0.0001}}}
    return black_bag_dic
