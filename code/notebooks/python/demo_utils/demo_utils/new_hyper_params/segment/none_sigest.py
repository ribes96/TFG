def get_segment_none_sigest_dic():
    none_sigest_dic = {
        'dt': {'identity': {'min_impurity_decrease': 0.0001},
               'rff': {'min_impurity_decrease': 0.001,
                       'gamma': 0.00031784090318295633},
               'nystroem': {'min_impurity_decrease': 0.0001,
                            'gamma': 0.00031784090318295633}},
        'logit': {'identity': {'C': 1000},
                  'rff': {'C': 1000,
                          'gamma': 0.00031784090318295633},
                  'nystroem': {'C': 1000,
                               'gamma': 0.00031784090318295633}},
        'linear_svc': {'identity': {'C': 1},
                       'rff': {'C': 100,
                               'gamma': 0.00031784090318295633},
                       'nystroem': {'C': 100,
                                    'gamma': 0.00031784090318295633}}}
    return none_sigest_dic
