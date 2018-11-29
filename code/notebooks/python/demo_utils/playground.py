# Ready to wipe

# # Aquí haré las pruebas que más adelante podrán ir en la versión final
# r = {
#     'entero': (int, float),
#     'palabra': str,
#     'otra_palabra': str,
#     'otro_entero': int
# }
#
#
# def check_types(good, test):
#     '''
#     Checks test to see if all variables have the corret type. If they not, an
#     exception is raised
#
#     Parameters
#     -----------
#     - good: a dictionary[str: type], where the str is the name of the
#       variable
#     to check. type can be a tupple of types
#     - test: a dictionary[str: value] to test. Both dictionarys are expected
#       to
#     share the keys
#
#     Returns
#     -------
#     None
#     '''
#
#     for k in test:
#         if not isinstance(test[k], good[k]):
#             raise ValueError('Wrong type for {0}: {1} was given \
#             and {2} was expected'.format(k, type(test[k]), good[k]))
#
#
# def f(entero=None, palabra=None, otra_palabra=None, otro_entero=None):
#     params = locals()
#     print(params)
#     check_types(r, params)
#     print("Se ha hecho el checkeo")
#
#
# # def get_non_sampling_model_scores(clf, dataset):
# #     '''
# #     Assuming clf is a model which DOESN'T use sampling, get the scores for
#       a
# #     given dataset
# #
# #     Parameters
# #     ----------
# #     clf : abstract model
# #         needs to implement fit and score, like scikit-learn
# #     dataset : dict
# #         Required keys: ['data_train', 'data_test', 'target_train',
# #         'target_test']
# #
# #     Returns
# #     -------
# #     tuple of float
# #         (train_score, test_score)
# #     '''
# #     data_train = dataset['data_train']
# #     data_test = dataset['data_test']
# #     target_train = dataset['target_train']
# #     target_test = dataset['target_test']
# #
# #     clf.fit(data_train, target_train)
# #
# #     train_score = clf.score(data_train, target_train)
# #     test_score = clf.score(data_test, target_test)
# #
# #     return train_score, test_score
#
#
# # def get_sampling_model_scores(clf, dataset, features):
# #     '''
# #     Assuming clf is a model which DO use sampling, get the scores for a
#       given
# #     dataset
# #
# #     Parameters
# #     ----------
# #     clf : abstract model
# #         needs to implement set_params(sampler__n_components=f) and fit()
#           and
# #         score(), like scikit-learn
# #     dataset : dict
# #         Required keys: ['data_train', 'data_test', 'target_train',
# #         'target_test']
# #     features : list of int
# #         The features on which to test
# #
# #     Returns
# #     -------
# #     tuple of dict
# #         (train, test), with keys ['ord', 'absi']
# #     '''
# #
# #     data_train = dataset['data_train']
# #     data_test = dataset['data_test']
# #     target_train = dataset['target_train']
# #     target_test = dataset['target_test']
# #
# #     train_scores = []
# #     test_scores = []
# #     for f in features:
# #         clf.set_params(sampler__n_components=f)
# #         clf.fit(data_train, target_train)
# #         train_score = clf.score(data_train, target_train)
# #         test_score = clf.score(data_test, target_test)
# #
# #         train_scores.append(train_score)
# #         test_scores.append(test_score)
# #
# #     train_dic = {
# #         'absi': features,
# #         'ord': train_scores,
# #     }
# #     test_dic = {
# #         'absi': features,
# #         'ord': test_scores,
# #     }
# #     return train_dic, test_dic
