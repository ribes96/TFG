
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted
from .ribes_RFFSampler import ribes_RFFSampler


class ribes_DecisionTreeClassifier_method2(DecisionTreeClassifier):
    """It is just as DecisionTreeClassifier from scikit-learn library (see
    http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
    but it will transform the input data with ribes_RFFSampler

    The parameters are the same as DecisionTreeClassifier plus n_RFF, which
    is an integer indicating the number of RFF. Remember that if n_RFF is n,
    then the resulting mapping will have 2*n columns, n for sin and n for cos

    If n_RFF is None, then it will be the number of features in the input data
    """
    def __init__(self,
                 criterion="gini",
                 splitter="best",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features=None,
                 random_state=None,
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 class_weight=None,
                 presort=False,

                 n_RFF=None):
        super(ribes_DecisionTreeClassifier_method2, self).__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            class_weight=class_weight,
            random_state=random_state,
            min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split,
            presort=presort)

        self.n_RFF = n_RFF
        print("ribes: ribes_DecisionTreeClassifier_method2 has been created")

    def fit(self, X, y, sample_weight=None, check_input=True, X_idx_sorted=None):
        print("ribes: empieza fit de ribes_DecisionTreeClassifier_method2")
        n_RFF = self.n_RFF
        if n_RFF is None:
            n_RFF = X.shape[1]
        sampler = ribes_RFFSampler(n_components = n_RFF)
        X = sampler.fit_transform(X)
        super(ribes_DecisionTreeClassifier_method2, self).fit(X,
                                                              y,
                                                              sample_weight=sample_weight,
                                                              check_input=check_input,
                                                              X_idx_sorted=X_idx_sorted)
        return self

    def apply(self, X, check_input = True):
        print("ribes: empieza apply de ribes_DecisionTreeClassifier_method2")
        n_RFF = self.n_RFF
        if n_RFF is None:
            n_RFF = X.shape[1]
        sampler = ribes_RFFSampler(n_components = n_RFF)
        X = sampler.fit_transform(X)
        return super(ribes_DecisionTreeClassifier_method2, self).apply(X,
                                                                       check_input = check_input)

    def decision_path(self, X, check_input=True):
        print("ribes: empieza decision_path de ribes_DecisionTreeClassifier_method2")
        n_RFF = self.n_RFF
        if n_RFF is None:
            n_RFF = X.shape[1]
        sampler = ribes_RFFSampler(n_components = n_RFF)
        X = sampler.fit_transform(X)
        return super(ribes_DecisionTreeClassifier_method2, self).decision_path(X,
                                                                               check_input=check_input)

    # This method is needed, since it does NOT call predict_proba nor any other
    # modified method, so it needs to transform the input data
    def predict(self, X, check_input=True):
        print("ribes: empieza predict de ribes_DecisionTreeClassifier_method2")
        n_RFF = self.n_RFF
        if n_RFF is None:
            n_RFF = X.shape[1]
        sampler = ribes_RFFSampler(n_components = n_RFF)
        X = sampler.fit_transform(X)
        return super(ribes_DecisionTreeClassifier_method2, self).predict(X,
                                                                         check_input=check_input)

    # predict_proba does NOT call predict, since it calls directly the predict
    # method of the internal tree, _tree, just like predict from the general tree
    def predict_proba(self, X, check_input=True):
        print("ribes: empieza predict_proba de ribes_DecisionTreeClassifier_method2")
        n_RFF = self.n_RFF
        if n_RFF is None:
            n_RFF = X.shape[1]
        sampler = ribes_RFFSampler(n_components = n_RFF)
        X = sampler.fit_transform(X)
        return super(ribes_DecisionTreeClassifier_method2, self).predict_proba(X)

    # There is no need to implement score, since it calls to method predict
    # The method score is implemented in the superclass ClassifierMixin, which
    # calls its own method predict, which is the one from the tree, since
    # the instance is of class tree

    # There isn't also the need to rewrite predict_log_proba, since it call
    # predict_proba
