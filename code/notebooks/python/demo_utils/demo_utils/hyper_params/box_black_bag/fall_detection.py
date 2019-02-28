#!/usr/bin/python3

##########################
# dt
##########################


def fall_detection_dic():
    dt = {
        'no_pca': {'min_impurity_decrease': 0.0001},
        'pca':  {'min_impurity_decrease': 0.001},
    }

    dt_rff = {
        'no_pca': {
            'gamma': 0.0001,
            'min_impurity_decrease': 1e-5,
        },
        'pca_first': {
            'gamma': 0.1,
            'min_impurity_decrease': 1e-7,
        },
        'pca_last': {
            'gamma': 1,
            'min_impurity_decrease': 1e-10,
        },
    }

    dt_nystroem = {
        'no_pca': {
            'gamma': 0.001,
            'min_impurity_decrease': 1e-9,
        },
        'pca_first': {
            'gamma': 0.001,
            'min_impurity_decrease': 1e-8,
        },
        'pca_last': {
            'gamma': 1,
            'min_impurity_decrease': 1e-5,
        },
    }

    ##########################
    # logit
    ##########################

    logit = {
        'no_pca': {'C': 0.001},
        'pca':  {'C': 0.001},
    }

    logit_rff = {
        'no_pca': {
            'gamma': 10,
            'C': 100,
        },
        'pca_first': {
            'gamma': 10,
            'C': 10,
        },
        'pca_last': {
            'gamma': 10,
            'C': 10,
        },
    }

    logit_nystroem = {
        'no_pca': {
            'gamma': 10,
            'C': 1000,
        },
        'pca_first': {
            'gamma': 10,
            'C': 10,
        },
        'pca_last': {
            'gamma': 10,
            'C': 10,
        },
    }

    ##########################
    # linear_svc
    ##########################

    linear_svc = {
        'no_pca': {'C': 0.001},
        'pca':  {'C': 0.001},
    }

    linear_svc_rff = {
        'no_pca': {
            'gamma': 1,
            'C': 100,
        },
        'pca_first': {
            'gamma': 10,
            'C': 1,
        },
        'pca_last': {
            'gamma': 10,
            'C': 10,
        },
    }

    linear_svc_nystroem = {
        'no_pca': {
            'gamma': 1,
            'C': 100,
        },
        'pca_first': {
            'gamma': 10,
            'C': 1,
        },
        'pca_last': {
            'gamma': 10,
            'C': 1,
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
