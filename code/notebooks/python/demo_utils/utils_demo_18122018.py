## General data
rang = (50, 800)
siz = 2000




l = {
    'model_name': 'logit',
    'sampler_name': 'identity',
    'box_type': 'none',
    'n_estim': None,
    'pca': False
}
l_rbf_b = {
    'model_name': 'logit',
    'sampler_name': 'rbf',
    'box_type': 'black',
    'n_estim': 20,
    'pca': False
}
l_rbf_g = {
    'model_name': 'logit',
    'sampler_name': 'rbf',
    'box_type': 'grey',
    'n_estim': 20,
    'pca': False
}

# Con PCA
l_pca = {
    'model_name': 'logit',
    'sampler_name': 'identity',
    'box_type': 'none',
    'n_estim': None,
    'pca': True
}
l_rbf_b_pca = {
    'model_name': 'logit',
    'sampler_name': 'rbf',
    'box_type': 'black',
    'n_estim': 20,
    'pca': True
}
l_rbf_g_pca = {
    'model_name': 'logit',
    'sampler_name': 'rbf',
    'box_type': 'grey',
    'n_estim': 20,
    'pca': True
}




####################
## Info
####################
info_l_rbf_mnist = {
    'models': [l, l_rbf_b, l_rbf_g],
    'features_range': rang,
    'dts_size': siz,
    'dts_name': 'mnist'
}

info_l_rbf_segment = {
    'models': [l, l_rbf_b, l_rbf_g],
    'features_range': rang,
    'dts_size': siz,
    'dts_name': 'segment'
}


info_l_rbf_mnist_pca = {
    'models': [l_pca, l_rbf_b_pca, l_rbf_g_pca],
    'features_range': rang,
    'dts_size': siz,
    'dts_name': 'mnist'
}

info_l_rbf_segment_pca = {
    'models': [l_pca, l_rbf_b_pca, l_rbf_g_pca],
    'features_range': rang,
    'dts_size': siz,
    'dts_name': 'segment'
}


###############################################
########### Con linear_svc
#######################################3

svc = {
    'model_name': 'linear_svc',
    'sampler_name': 'identity',
    'box_type': 'none',
    'n_estim': None,
    'pca': False
}
svc_rbf_b = {
    'model_name': 'linear_svc',
    'sampler_name': 'rbf',
    'box_type': 'black',
    'n_estim': 20,
    'pca': False
}
svc_rbf_g = {
    'model_name': 'linear_svc',
    'sampler_name': 'rbf',
    'box_type': 'grey',
    'n_estim': 20,
    'pca': False
}

# Con PCA
svc_pca = {
    'model_name': 'linear_svc',
    'sampler_name': 'identity',
    'box_type': 'none',
    'n_estim': None,
    'pca': True
}
svc_rbf_b_pca = {
    'model_name': 'linear_svc',
    'sampler_name': 'rbf',
    'box_type': 'black',
    'n_estim': 20,
    'pca': True
}
svc_rbf_g_pca = {
    'model_name': 'linear_svc',
    'sampler_name': 'rbf',
    'box_type': 'grey',
    'n_estim': 20,
    'pca': True
}




####################
## Info
####################
info_svc_rbf_mnist = {
    'models': [svc, svc_rbf_b, svc_rbf_g],
    'features_range': rang,
    'dts_size': siz,
    'dts_name': 'mnist'
}

info_svc_rbf_segment = {
    'models': [svc, svc_rbf_b, svc_rbf_g],
    'features_range': rang,
    'dts_size': siz,
    'dts_name': 'segment'
}


info_svc_rbf_mnist_pca = {
    'models': [svc_pca, svc_rbf_b_pca, svc_rbf_g_pca],
    'features_range': rang,
    'dts_size': siz,
    'dts_name': 'mnist'
}

info_svc_rbf_segment_pca = {
    'models': [svc_pca, svc_rbf_b_pca, svc_rbf_g_pca],
    'features_range': rang,
    'dts_size': siz,
    'dts_name': 'segment'
}
