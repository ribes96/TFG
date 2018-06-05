# todo Todavía no está hecho. Simplemente tiene que entender los votos de los
# ártboles como hard
# Cuando esté terminado tengo que hacer que las otras tres clases hereden de
# esta

from sklearn.ensemble import RandomForestClassifier
class ribes_BaseRandomForestClassifier(RandomForestClassifier):
    """A RandomForestClassifier.

    It perform almost exactly the same thing as RandomForestClassifier, from
    the scikit-learn library (http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
    The only difference is that the predicted class will be the one which is the
    most probable by the majority of the trees, instead of using the mean
    probability estimate across the trees.
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
                 class_weight=None):
        super(ribes_BaseRandomForestClassifier, self).__init__(
             n_estimators=n_estimators,
             criterion=criterion,
             max_depth=max_depth,
             min_samples_split=min_samples_split,
             min_samples_leaf=min_samples_leaf,
             min_weight_fraction_leaf=min_weight_fraction_leaf,
             max_features=max_features,
             max_leaf_nodes=max_leaf_nodes,
             min_impurity_decrease=min_impurity_decrease,
             min_impurity_split=min_impurity_split,
             bootstrap=bootstrap,
             oob_score=oob_score,
             n_jobs=n_jobs,
             random_state=random_state,
             verbose=verbose,
             warm_start=warm_start,
             class_weight=class_weight)

    def predict_proba(X):
        """
        """
        print("Hola, estamos en predict_proba")
        check_is_fitted(self, 'estimators_')
        # Check data
        X = self._validate_X_predict(X)

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
