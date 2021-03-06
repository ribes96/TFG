
def get_mnist_none_dic():
    none_dic = {'dt': {'no_sampler': {'no_pca': {'min_impurity_decrease': 1e-05},
                                      'pca': {'min_impurity_decrease': 0.01}},
                       'rff': {'no_pca': {'min_impurity_decrease': 1e-06, 'gamma': 1e-05},
                               'pca_first': {'min_impurity_decrease': 0.0001, 'gamma': 1e-05},
                               'pca_last': {'min_impurity_decrease': 0.01, 'gamma': 0.0001}},
                       'nystroem': {'no_pca': {'min_impurity_decrease': 1e-08, 'gamma': 0.001},
                                    'pca_first': {'min_impurity_decrease': 0.0001, 'gamma': 0.0001},
                                    'pca_last': {'min_impurity_decrease': 1e-06, 'gamma': 0.001}}},
                'logit': {'no_sampler': {'no_pca': {'C': 0.01}, 'pca': {'C': 0.01}},
                          'rff': {'no_pca': {'C': 1000, 'gamma': 1e-05},
                                  'pca_first': {'C': 100, 'gamma': 0.0001},
                                  'pca_last': {'C': 100, 'gamma': 0.0001}},
                          'nystroem': {'no_pca': {'C': 100, 'gamma': 0.0001},
                                       'pca_first': {'C': 1000, 'gamma': 1e-05},
                                       'pca_last': {'C': 100, 'gamma': 0.0001}}},
                'linear_svc': {'no_sampler': {'no_pca': {'C': 0.0001}, 'pca': {'C': 0.0001}},
                               'rff': {'no_pca': {'C': 100, 'gamma': 1e-05},
                                       'pca_first': {'C': 100, 'gamma': 1e-05},
                                       'pca_last': {'C': 10, 'gamma': 0.0001}},
                               'nystroem': {'no_pca': {'C': 10, 'gamma': 0.0001},
                                            'pca_first': {'C': 1, 'gamma': 0.001},
                                            'pca_last': {'C': 10, 'gamma': 0.001}}}}
    return none_dic
