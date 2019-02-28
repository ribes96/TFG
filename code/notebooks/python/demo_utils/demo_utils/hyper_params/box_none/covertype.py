#!/usr/bin/python3

##########################
# dt
##########################


def covertype_dic():
    dt = {
        'no_pca': {'min_impurity_decrease': 0.0001},
        'pca':  {'min_impurity_decrease': 0.001},
    }

    dt_rff = {
        'no_pca': {
            'gamma': 0.0001,  # X
            'min_impurity_decrease': 1e-7,  # X
        },
        'pca_first': {
            'gamma': 0.001,  # X
            'min_impurity_decrease': 1e-9,  # X
        },
        'pca_last': {
            'gamma': 0.001,  # X
            'min_impurity_decrease': 1e-10,  # X
        },
    }

    dt_nystroem = {
        'no_pca': {
            'gamma': 0.01,  # X
            'min_impurity_decrease': 1e-5,  # X
        },
        'pca_first': {
            'gamma': 0.01,  # X
            'min_impurity_decrease': 1e-10,  # X
        },
        'pca_last': {
            'gamma': 1e-5,  # X
            'min_impurity_decrease': 1e-10,  # X
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
            'gamma': 0.01,  # X
            'C': 1000,  # X
        },
        'pca_first': {
            'gamma': 0.01,  # X
            'C': 100,  # X
        },
        'pca_last': {
            'gamma': 0.01,  # X
            'C': 1000,  # X
        },
    }

    logit_nystroem = {
        'no_pca': {
            'gamma': 0.1,  # X
            'C': 100,  # X
        },
        'pca_first': {
            'gamma': 0.1,  # X
            'C': 100,  # X
        },
        'pca_last': {
            'gamma': 0.01,  # X
            'C': 1000,  # X
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
            'gamma': 0.1,  # X
            'C': 1,  # X
        },
        'pca_first': {
            'gamma': 0.1,  # X
            'C': 10,  # X
        },
        'pca_last': {
            'gamma': 0.1,  # X
            'C': 1,  # X
        },
    }

    linear_svc_nystroem = {
        'no_pca': {
            'gamma': 0.1,  # X
            'C': 10,  # X
        },
        'pca_first': {
            'gamma': 0.1,  # X
            'C': 100,  # X
        },
        'pca_last': {
            'gamma': 0.1,  # X
            'C': 1,  # X
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
