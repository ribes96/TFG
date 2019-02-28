m1 = {
    'model_name': 'dt',
    'sampler_name': 'identity',
    'box_type': 'none',
    'n_estim': None,
    'pca': False
}

m2 = {
    'model_name': 'dt',
    'sampler_name': 'identity',
    'box_type': 'none',
    'n_estim': None,
    'pca': True
}

m3 = {
    'model_name': 'dt',
    'sampler_name': 'rbf',
    'box_type': 'none',
    'n_estim': None,
    'pca': True
}

m4 = {
    'model_name': 'dt',
    'sampler_name': 'rbf',
    'box_type': 'none',
    'n_estim': None,
    'pca': False
}

m5 = {
    'model_name': 'dt',
    'sampler_name': 'nystroem',
    'box_type': 'none',
    'n_estim': None,
    'pca': False
}

m6 = {
    'model_name': 'dt',
    'sampler_name': 'nystroem',
    'box_type': 'none',
    'n_estim': None,
    'pca': True
}

data_d0 = {
    'models': [m1, m2, m3, m4, m5, m6],
    'features_range': (30, 600),
    'dts_size': 2000,
    'dts_name': 'digits'}

data_d1_1 = {
    'features_range': (30, 600),
    'model_data': {
        'model_name': 'dt',
        'sampler_name': 'rbf',
        'n_estim': None,
        'box_type': 'none'},
    'dts_size': 2000,
    'dts_name': 'digits'}

data_d1_2 = {
    'features_range': (30, 600),
    'model_data': {
        'model_name': 'dt',
        'sampler_name': 'nystroem',
        'n_estim': None,
        'box_type': 'none'},
    'dts_size': 2000,
    'dts_name': 'digits'}


w1 = {'model_name': 'logit', 'sampler_name': 'identity', 'box_type': 'none', 'n_estim': None, 'pca': False}
w2 = {'model_name': 'logit', 'sampler_name': 'identity', 'box_type': 'none', 'n_estim': None, 'pca': True}
w3 = {'model_name': 'logit', 'sampler_name': 'rbf', 'box_type': 'none', 'n_estim': None, 'pca': False}
w4 = {'model_name': 'logit', 'sampler_name': 'rbf', 'box_type': 'none', 'n_estim': None, 'pca': True}
w5 = {'model_name': 'logit', 'sampler_name': 'rbf', 'box_type': 'black', 'n_estim': 30, 'pca': False}
w6 = {'model_name': 'logit', 'sampler_name': 'rbf', 'box_type': 'black', 'n_estim': 30, 'pca': True}
w7 = {'model_name': 'logit', 'sampler_name': 'rbf', 'box_type': 'grey', 'n_estim': 30, 'pca': False}
w8 = {'model_name': 'logit', 'sampler_name': 'rbf', 'box_type': 'grey', 'n_estim': 30, 'pca': True}

data_d2_3 = {
    'models': [w1, w2, w3, w4, w5, w6, w7, w8],
    'features_range': (30, 600),
    'dts_size': 2000,
    'dts_name': 'segment'}

data_d2_1 = {
    'models': [w1, w2, w3, w4, w5, w6, w7, w8],
    'features_range': (30, 600),
    'dts_size': 2000,
    'dts_name': 'digits'}

w1 = {'model_name': 'logit', 'sampler_name': 'identity', 'box_type': 'none', 'n_estim': None, 'pca': False}
w2 = {'model_name': 'logit', 'sampler_name': 'identity', 'box_type': 'none', 'n_estim': None, 'pca': True}
w3 = {'model_name': 'logit', 'sampler_name': 'nystroem', 'box_type': 'none', 'n_estim': None, 'pca': False}
w4 = {'model_name': 'logit', 'sampler_name': 'nystroem', 'box_type': 'none', 'n_estim': None, 'pca': True}
w5 = {'model_name': 'logit', 'sampler_name': 'nystroem', 'box_type': 'black', 'n_estim': 30, 'pca': False}
w6 = {'model_name': 'logit', 'sampler_name': 'nystroem', 'box_type': 'black', 'n_estim': 30, 'pca': True}
w7 = {'model_name': 'logit', 'sampler_name': 'nystroem', 'box_type': 'grey', 'n_estim': 30, 'pca': False}
w8 = {'model_name': 'logit', 'sampler_name': 'nystroem', 'box_type': 'grey', 'n_estim': 30, 'pca': True}

data_d2_4 = {
    'models': [w1, w2 ,w3 ,w4 ,w5 ,w6 ,w7 ,w8 ],
    'features_range': (30, 600),
    'dts_size': 2000,
    'dts_name': 'segment'}

data_d2_2 = {
    'models': [w1, w2 ,w3 ,w4 ,w5 ,w6 ,w7 ,w8 ],
    'features_range': (30, 600),
    'dts_size': 2000,
    'dts_name': 'digits'}
