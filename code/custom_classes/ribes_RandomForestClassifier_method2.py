
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_is_fitted
from sklearn.ensemble.base import _partition_estimators
import numpy as np
import threading
from sklearn.externals.joblib import Parallel, delayed

from .ribes_DecisionTreeClassifier_method2 import ribes_DecisionTreeClassifier_method2

from .ribes_RFFSampler import ribes_RFFSampler



# This is a utility function for joblib's Parallel. It can't go locally in
# ForestClassifier or ForestRegressor, because joblib complains that it cannot
# pickle it when placed there.

def accumulate_prediction(predict, X, out, lock):
    prediction = predict(X, check_input=False)
    with lock:
        if len(out) == 1:
            out[0] += prediction
        else:
            for i in range(len(out)):
                out[i] += prediction[i]

class ribes_RandomForestClassifier_method2(RandomForestClassifier):
    """A RandomForestClassifier which trains each of the trees of the forest
    with a different mapping of the data.

    The algorithm is exactly the same as the original RandomForestClassifier. It
    just feeds each of the trees with a different mapping

    The added parameter is n_RFF. If None, is the number of
    features of the data
    Remember the if you say n features, the resulting mapping will have 2 * n
    columns, n for sin and n for cos
    """
    def __init__(self,
                 n_estimators=10,
                 criterion="gini",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 class_weight=None,

                 n_RFF = None
                 ):
        # Caution! I have called the father of RandomForestClassifier since I
        # need to build it with a different type of tree
        # The only change is the base_estimator parameter
        # There is also de self.n_RFF, for later use (in predict_proba, not sure
        # if it is really needed)
        super(RandomForestClassifier, self).__init__(
            base_estimator=ribes_DecisionTreeClassifier_method2(n_RFF=n_RFF),
            n_estimators=n_estimators,
            estimator_params=("criterion", "max_depth", "min_samples_split",
                              "min_samples_leaf", "min_weight_fraction_leaf",
                              "max_features", "max_leaf_nodes",
                              "min_impurity_decrease", "min_impurity_split",
                              "random_state"),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight)

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split

        # Just for the fake _validate_X_predict
        self.n_RFF = n_RFF


    # The only reason to rewrite this function is to avoid the  _validate_X_predict
    # call from failing the whole program. It won't pass the test due to the
    # changes in the input data
    # The function accumulate_prediction has been writen in the top
    def predict_proba(self, X):
        print("ribes: empieza predict_proba de ribes_RandomForestClassifier_method2")
        #############################
        n_RFF = self.n_RFF
        if n_RFF is None:
            n_RFF = X.shape[1]
        sampler = ribes_RFFSampler(n_components = n_RFF)
        junk = sampler.fit_transform(X)
        #################################
        check_is_fitted(self, 'estimators_')
        # Check data
        junk = self._validate_X_predict(junk)

        # Assign chunk of trees to jobs
        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)

        # avoid storing the output of every estimator by summing them here
        all_proba = [np.zeros((X.shape[0], j), dtype=np.float64)
                     for j in np.atleast_1d(self.n_classes_)]
        lock = threading.Lock()
        Parallel(n_jobs=n_jobs, verbose=self.verbose, backend="threading")(
            delayed(accumulate_prediction)(e.predict_proba, X, all_proba, lock)
            for e in self.estimators_)

        for proba in all_proba:
            proba /= len(self.estimators_)

        if len(all_proba) == 1:
            return all_proba[0]
        else:
            return all_proba
