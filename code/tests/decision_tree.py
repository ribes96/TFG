#!/usr/bin/python3

import sys
sys.path.append("..")
# sys.path.append("..")
from sklearn.tree import DecisionTreeClassifier
from custom_classes.ribes_RFFSampler import ribes_RFFSampler
from sklearn.kernel_approximation import RBFSampler


from read_data import get_data

train_data, train_predictions, test_data, test_predictions = get_data()
train_predictions, test_predictions = train_predictions.ravel(), test_predictions.ravel()

desired_components = 16

sampler = RBFSampler(n_components = desired_components)
sampler.fit(train_data)
mapped_train_data = sampler.transform(train_data)
mapped_test_data = sampler.transform(test_data)
arbol = DecisionTreeClassifier()
arbol.fit(mapped_train_data, train_predictions)
test_score = arbol.score(mapped_test_data, test_predictions)
print(test_score)



# print(train_data[1:10, :])
# print(mapped_train_data[1:10,:])

# sampler = RBFSampler(n_components = desired_components)
# sampler.fit(train_data)
# # RBFSampler_train_data = sampler.fit_transform(train_data)
# RBFSampler_train_data = sampler.transform(train_data)
#
# # RBFSampler_test_data = sampler.fit_transform(test_data)
# RBFSampler_test_data = sampler.transform(test_data)
#
# # sampler = sampler.fit(train_data)
# # train_data2 = sampler.transform(train_data)
# # test_data2 = sampler.transform(test_data)
#
#
# ribes_sampler = ribes_RFFSampler(n_components = desired_components)
# ribes_sampler.fit(train_data)
# # ribes_train_data = ribes_sampler.fit_transform(train_data)
# ribes_train_data = ribes_sampler.transform(train_data)
# # ribes_test_data = ribes_sampler.fit_transform(test_data)
# ribes_test_data = ribes_sampler.transform(test_data)
#
#
# # Train and score
# orig_tree = DecisionTreeClassifier()
# orig_tree.fit(train_data, train_predictions.ravel())
# orig_tree_score = orig_tree.score(test_data, test_predictions.ravel())
# # orig_tree_score = orig_tree.score(train_data, train_predictions.ravel())
#
# orig_tree.fit(RBFSampler_train_data, train_predictions.ravel())
# # RBFSampler_score = orig_tree.score(RBFSampler_test_data, test_predictions.ravel())
# RBFSampler_score = orig_tree.score(RBFSampler_train_data, train_predictions.ravel())
#
# # orig_tree.fit(train_data2, train_predictions.ravel())
# # score2 = orig_tree.score(test_data2, test_predictions.ravel())
#
# orig_tree.fit(ribes_train_data, train_predictions.ravel())
# ribes_score = orig_tree.score(ribes_test_data, test_predictions.ravel())
#
# print("--------------------------------------------------------------------")
# print()
# print("Accuracy using original data:")
# print(orig_tree_score)
#
# print()
# print("Accuracy using the RBFSampler with different fits:")
# print(RBFSampler_score)
#
# print()
# print("Accuracy using the RBFSampler with the same fit:")
# print(score2)
#
# print()
# print("Accuracy using the ribes_RFFSampler with the same fit:")
# print(ribes_score)
# #
