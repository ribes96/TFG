#!/usr/bin/python3

##########################
# dt
##########################


def vowel_dic():
    dt = {
        'no_pca': {'min_impurity_decrease': 0.0001},
        'pca':  {'min_impurity_decrease': 0.001},
    }

    dt_rff = {
        'no_pca': {
            'gamma': 4,
            'min_impurity_decrease': 0.001,
        },
        'pca_first': {
            'gamma': 4,
            'min_impurity_decrease': 0.001,
        },
        'pca_last': {
            'gamma': 4,
            'min_impurity_decrease': 0.001,
        },
    }

    dt_nystroem = {
        'no_pca': {
            'gamma': 4,
            'min_impurity_decrease': 0.001,
        },
        'pca_first': {
            'gamma': 4,
            'min_impurity_decrease': 0.001,
        },
        'pca_last': {
            'gamma': 4,
            'min_impurity_decrease': 0.001,
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
            'gamma': 4,
            'C': 0.001,
        },
        'pca_first': {
            'gamma': 4,
            'C': 0.001,
        },
        'pca_last': {
            'gamma': 4,
            'C': 0.001,
        },
    }

    logit_nystroem = {
        'no_pca': {
            'gamma': 4,
            'C': 0.001,
        },
        'pca_first': {
            'gamma': 4,
            'C': 0.001,
        },
        'pca_last': {
            'gamma': 4,
            'C': 0.001,
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
            'gamma': 4,
            'C': 0.001,
        },
        'pca_first': {
            'gamma': 4,
            'C': 0.001,
        },
        'pca_last': {
            'gamma': 4,
            'C': 0.001,
        },
    }

    linear_svc_nystroem = {
        'no_pca': {
            'gamma': 4,
            'C': 0.001,
        },
        'pca_first': {
            'gamma': 4,
            'C': 0.001,
        },
        'pca_last': {
            'gamma': 4,
            'C': 0.001,
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
