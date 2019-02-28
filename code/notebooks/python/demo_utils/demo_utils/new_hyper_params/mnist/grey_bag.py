def get_mnist_grey_bag_dic():
    grey_bag_dic = {'dt': {'identity': {'min_impurity_decrease': 0},
                           'rff': {'min_impurity_decrease': 0,
                                   'gamma': 0.0001},
                           'nystroem': {'min_impurity_decrease': 0,
                                        'gamma': 0.001}},
                    'logit': {'identity': {'C': 1000},
                              'rff': {'C': 1000, 'gamma': 0.0001},
                              'nystroem': {'C': 1000, 'gamma': 0.001}},
                    'linear_svc': {'identity': {'C': 1000},
                                   'rff': {'C': 1000, 'gamma': 0.0001},
                                   'nystroem': {'C': 1000, 'gamma': 0.001}}}
    return grey_bag_dic
