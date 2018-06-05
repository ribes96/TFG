

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from .ribes_RFFSampler import ribes_RFFSampler



class ribes_RandomForestClassifier_method1(RandomForestClassifier):
    """A RandomForestClassifier which uses a RFF mapping of the data instead
    of directly using the data

    The algorithm is exactly the same as the original RandomForestClassifier. It
    just changes the original data

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
        super(ribes_RandomForestClassifier_method1, self).__init__(
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

        self.n_RFF = n_RFF

    def fit(self, X, y, sample_weight=None):
        print("ribes: empieza fit de ribes_RandomForestClassifier_method1")
        n_RFF = self.n_RFF
        if n_RFF is None:
            n_RFF = X.shape[1]
        sampler = ribes_RFFSampler(n_components = n_RFF)
        X = sampler.fit_transform(X)
        super(ribes_RandomForestClassifier_method1, self).fit(X,
                                                              y,
                                                              sample_weight=sample_weight)
        return self

    def apply(self, X):
        print("ribes: empieza apply de ribes_RandomForestClassifier_method1")
        n_RFF = self.n_RFF
        if n_RFF is None:
            n_RFF = X.shape[1]
        sampler = ribes_RFFSampler(n_components = n_RFF)
        X = sampler.fit_transform(X)
        return super(ribes_RandomForestClassifier_method1, self).apply(X)

    def decision_path(self, X):
        print("ribes: empieza decision_path de ribes_RandomForestClassifier_method1")
        n_RFF = self.n_RFF
        if n_RFF is None:
            n_RFF = X.shape[1]
        sampler = ribes_RFFSampler(n_components = n_RFF)
        X = sampler.fit_transform(X)
        return super(ribes_RandomForestClassifier_method1, self).decision_path(X)

    def predict_proba(self, X):
        print("ribes: empieza predict_proba de ribes_RandomForestClassifier_method1")
        n_RFF = self.n_RFF
        if n_RFF is None:
            n_RFF = X.shape[1]
        sampler = ribes_RFFSampler(n_components = n_RFF)
        X = sampler.fit_transform(X)
        return super(ribes_RandomForestClassifier_method1, self).predict_proba(X)

    # There is no need to implement predict and predict_log_proba, since they
     # call to method predict_proba
