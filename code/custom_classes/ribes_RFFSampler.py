import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.utils import check_array, check_random_state
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.preprocessing import normalize

class ribes_RFFSampler(RBFSampler):
    """Approximates feature map of an RBF kernel by Monte Carlo approximation
    of its Fourier transform.

    This is a custom class writen by Albert Ribes Marzá
    """

    def __init__(self, gamma=1., n_components=100, random_state=None):
        print("ribes: empieza __init__ de ribes_RFFSampler")
        super(ribes_RFFSampler, self).__init__(
            gamma=gamma,
            n_components=n_components,
            random_state=random_state)
        # print("Y este soy yo")

    def fit(self, X, y=None):
        print("ribes: empieza fit de ribes_RFFSampler")
        X = normalize(X, norm = 'l2', axis = 1)
        super(ribes_RFFSampler, self).fit(X, y)
        return self
    #     X = check_array(X, accept_sparse='csr')
    #     random_state = check_random_state(self.random_state)
    #     n_features = X.shape[1]
    #
    #     self.random_weights_ = (np.sqrt(2 * self.gamma) * random_state.normal(
    #         size=(n_features, self.n_components)))
    #
    #     # self.random_offset_ = random_state.uniform(0, 2 * np.pi, size=self.n_components)
    #     # Nosotros no queremos ese offset para nada
    #     self.random_offset_ = np.random.uniform(0, 0, size=self.n_components)
    #     return self
    def transform(self, X):
        print("ribes: empieza transform de ribes_RFFSampler")
        check_is_fitted(self, 'random_weights_')

        X = check_array(X, accept_sparse='csr')
        projection = safe_sparse_dot(X, self.random_weights_)
        # projection += self.random_offset_

        cosMat = np.cos(projection)
        sinMat = np.sin(projection)
        cosMat *= np.sqrt(2.) / np.sqrt(self.n_components)
        sinMat *= np.sqrt(2.) / np.sqrt(self.n_components)

        result = np.concatenate((sinMat, cosMat), axis = 1)
        # return projection
        return result
