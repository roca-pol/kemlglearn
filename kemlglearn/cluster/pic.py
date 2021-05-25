import numpy as np
from sklearn.cluster import KMeans
from sklearn.base import ClusterMixin, BaseEstimator
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils.validation import check_random_state


class PIC(ClusterMixin, BaseEstimator):
    def __init__(self, n_clusters, affinity='cosine', init='random',
                 gamma=1.0, degree=3, coef0=1, kernel_params=None, 
                 eps=None, max_iter=50, n_init=10, random_state=None):
        self.n_clusters = n_clusters
        self.affinity = affinity
        self.init = init
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.eps = eps
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state

    def fit(self, X):
        X = self._validate_data(X, accept_sparse=['csr', 'csc', 'coo'],
                                dtype=np.float64, ensure_min_samples=2)

        if self.eps is None:
            eps = 1e-5 / X.shape[0]
        else:
            eps = self.eps

        W, v = self._compute_norm_affinity_mat(X)

        random_state = check_random_state(self.random_state)
        if self.init == 'random':
            v = random_state.random(size=W.shape[0])

        v_old = v + 2 * eps
        delta_old = v - v_old
        
        for i in range(self.max_iter):
            wv = W @ v
            v = wv / np.abs(wv).sum()
            delta = v - v_old

            if np.all(np.abs(delta - delta_old) < eps):
                break

            delta_old = delta
            v_old = v
        
        self.kmeans_ = KMeans(n_clusters=self.n_clusters,
                              n_init=self.n_init,
                              random_state=random_state)
        self.labels_ = self.kmeans_.fit_predict(v.reshape(-1, 1))
        self.v_ = v
        self.affinity_matrix_ = W
        return self

    def _compute_norm_affinity_mat(self, X):
        if self.affinity == 'precomputed':
            A = X
        else:
            params = self.kernel_params
            if params is None:
                params = {}
            if not callable(self.affinity):
                params['gamma'] = self.gamma
                params['degree'] = self.degree
                params['coef0'] = self.coef0
            A = pairwise_kernels(X, metric=self.affinity,
                                 filter_params=True,
                                 **params)

        # A = np.empty((X.shape[0], X.shape[0]))
        # for i in range(X.shape[0]):
        #     A[i, i:] = sim(X[i][None, ...], X[i:])
        #     A[i:, i] = A[i, i:]
            
        # normalize
        D = A.sum(axis=1)
        D_inv = np.reciprocal(D)
        W = D_inv.reshape(-1, 1) * A  # same as multiplying D_inv @ A

        v0 = D / D.sum()
        return W, v0