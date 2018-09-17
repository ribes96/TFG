import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.utils import check_array, check_random_state
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.preprocessing import normalize

class ribes_tan_RFFSampler(RBFSampler):
    """Approximates feature map of an RBF kernel by Monte Carlo approximation
    of its Fourier transform, but uses tan function instead of cos

    This is a custom class writen by Albert Ribes Marzá
    """
    def __init__(self, gamma=1., n_components=100, random_state=None):
        super(ribes_tan_RFFSampler, self).__init__(
            gamma=gamma,
            n_components=n_components,
            random_state=random_state)

    def transform(self, X):
        """Apply the approximate feature map to X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            New data, where n_samples in the number of samples
            and n_features is the number of features.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """
        check_is_fitted(self, 'random_weights_')

        X = check_array(X, accept_sparse='csr')
        projection = safe_sparse_dot(X, self.random_weights_)
        projection += self.random_offset_
        np.tan(projection, projection)
        projection *= np.sqrt(2.) / np.sqrt(self.n_components)
        return projection
