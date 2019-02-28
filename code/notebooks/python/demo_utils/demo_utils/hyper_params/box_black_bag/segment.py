#!/usr/bin/python3

##########################
# dt
##########################


def segment_dic():
    dt = {
        'no_pca': {'min_impurity_decrease': 1e-09},
        'pca':  {'min_impurity_decrease': 1e-10},
    }

    dt_rff = {
        'no_pca': {
            'gamma': 0.0001,
            'min_impurity_decrease': 1e-10,
        },
        'pca_first': {
            'gamma': 0.0001,
            'min_impurity_decrease': 1e-5,
        },
        'pca_last': {
            'gamma': 0.01,
            'min_impurity_decrease': 1e-5,
        },
    }

    dt_nystroem = {
        'no_pca': {
            'gamma': 0.0001,
            'min_impurity_decrease': 1e-6,
        },
        'pca_first': {
            'gamma': 0.0001,
            'min_impurity_decrease': 0.001,
        },
        'pca_last': {
            'gamma': 0.01,
            'min_impurity_decrease': 1e-9,
        },
    }

    ##########################
    # logit
    ##########################

    logit = {
        'no_pca': {'C': 100},
        'pca':  {'C': 1000},
    }

    logit_rff = {
        'no_pca': {
            'gamma': 0.01,
            'C': 1000,
        },
        'pca_first': {
            'gamma': 0.1,
            'C': 1000,
        },
        'pca_last': {
            'gamma': 0.1,
            'C': 100,
        },
    }

    logit_nystroem = {
        'no_pca': {
            'gamma': 0.01,
            'C': 1000,
        },
        'pca_first': {
            'gamma': 0.1,
            'C': 1000,
        },
        'pca_last': {
            'gamma': 0.1,
            'C': 1000,
        },
    }

    ##########################
    # linear_svc
    ##########################

    linear_svc = {
        'no_pca': {'C': 1},
        'pca':  {'C': 10},
    }

    linear_svc_rff = {
        'no_pca': {
            'gamma': 0.01,
            'C': 100,
        },
        'pca_first': {
            'gamma': 0.1,
            'C': 100,
        },
        'pca_last': {
            'gamma': 0.1,
            'C': 10,
        },
    }

    linear_svc_nystroem = {
        'no_pca': {
            'gamma': 0.1,
            'C': 1000,
        },
        'pca_first': {
            'gamma': 0.1,
            'C': 1000,
        },
        'pca_last': {
            'gamma': 0.1,
            'C': 10,
        },
    }
    general = {
        'dt': {
            'no_sampler': dt,
            'rff': dt_rff,
            'nystroem': dt_nystroem,
        },
        'logit': {
            'no_sampler': logit,
            'rff': logit_rff,
            'nystroem': logit_nystroem,
        },
        'linear_svc': {
            'no_sampler': linear_svc,
            'rff': linear_svc_rff,
            'nystroem': linear_svc_nystroem,
        },
    }

    return general
