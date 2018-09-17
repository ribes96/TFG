#!/usr/bin/python3

# Se trata de imitar el ejemplo que hay en
# http://scikit-learn.org/stable/auto_examples/plot_kernel_approximation.html
# sobre el uso de RBFSampler

import sys
sys.path.append("..")
from read_data import get_data
from sklearn.kernel_approximation import RBFSampler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.utils.extmath import safe_sparse_dot
import numpy as np

def ribes_get_W_offset(data, gamma, D):
    d = data.shape[1]
    W = (np.sqrt(2 * gamma) * np.random.normal(
            size=(d, D)))
    offset = np.random.uniform(0, 2 * np.pi, size = D)
    return W, offset

def ribes_map(data, W, offset):
    projection = safe_sparse_dot(data, W)
    projection += offset
    np.cos(projection, projection)
    projection *= np.sqrt(2.) / np.sqrt(W.shape[1])
    return projection

# print("hola")
train_data, train_predictions, test_data, test_predictions = get_data()
# sampler = RBFSampler(gamma=1, random_state=1)

# sampler.fit(train_data)
gamma = 1
D = 100
W, offset = ribes_get_W_offset(data = train_data, gamma = gamma, D = D)
mapped_train_data = ribes_map(data = train_data, W = W, offset = offset)
mapped_test_data = ribes_map(data = test_data, W = W, offset = offset)
# mapped_train_data = sampler.transform(train_data)
# mapped_test_data = sampler.transform(test_data)

arbol = DecisionTreeClassifier()
# arbol = RandomForestClassifier()
# arbol = SGDClassifier()
arbol.fit(mapped_train_data, train_predictions.ravel())
score = arbol.score(mapped_test_data, test_predictions)
print(score)
