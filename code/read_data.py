#!/usr/bin/python3

# This script just trains a RandomForestClassifier with a handwriten digits
# dataset with about 7500 instances, and then tests them with a testing dataset

import numpy
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from custom_classes.ribes_DecisionTreeClassifier_method2 import ribes_DecisionTreeClassifier_method2
from custom_classes.ribes_RandomForestClassifier_method1 import ribes_RandomForestClassifier_method1
from custom_classes.ribes_RandomForestClassifier_method2 import ribes_RandomForestClassifier_method2


train_data_path = "data/pendigits.tra"
test_data_path = "data/pendigits.tes"
n_features = 16

train_raw_data = open(train_data_path, 'rt')
test_raw_data = open(test_data_path, 'rt')

train_all_data = numpy.loadtxt(train_raw_data, delimiter=",")
test_all_data = numpy.loadtxt(test_raw_data, delimiter=",")

train_data, train_predictions = numpy.split(train_all_data, indices_or_sections = [n_features], axis = 1)
test_data, test_predictions = numpy.split(test_all_data, indices_or_sections = [n_features], axis = 1)

# clf = RandomForestClassifier(n_estimators = 5)
# print("Training with", len(train_predictions), "instances")
# clf.fit(train_data, train_predictions.ravel())
# print("Testing with", len(test_data), "instances")
# result = clf.predict(test_data)
# correct_prediction = test_predictions.ravel() == result.ravel()
# number_correct = numpy.count_nonzero(correct_prediction)
# print("Number of correct predictions:", number_correct)
# # print("Percentage of correct predictions:", percent, "%")
# score = clf.score(test_data, test_predictions.ravel())
# print("Score:", score)
# print("Error:", 1 - score)
#
# print("Empiezo con el custom")
# print("-----------------------------------------")

# c_clf = ribes_DecisionTreeClassifier_method2(n_RFF = None)
# o_clf = DecisionTreeClassifier()
# c_clf.fit(train_data, train_predictions.ravel())
# o_clf.fit(train_data, train_predictions.ravel())
# pr = c_clf.score(train_data, train_predictions.ravel())
# print(pr)

c_clf = ribes_RandomForestClassifier_method2(n_estimators=500, n_RFF = 500)
# c_clf = ribes_DecisionTreeClassifier_method2(n_RFF = 4)
# o_clf = RandomForestClassifier()
c_clf.fit(train_data, train_predictions.ravel())
# o_clf.fit(train_data, train_predictions.ravel())
pr = c_clf.score(train_data, train_predictions.ravel())
# pr = c_clf.predict_proba(train_data)
print(pr)
