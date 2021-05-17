import numpy as np
from sklearn.cluster import KMeans
from sklearn.base import ClusterMixin, BaseEstimator
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils.validation import check_random_state


class PIC(ClusterMixin, BaseEstimator):
    def __init__(self, n_clusters, clusterer=None, affinity='cosine', 
                 gamma=1.0, degree=3, coef0=1, kernel_params=None, 
                 eps=None, max_iter=50, random_state=None):
        self.n_clusters = n_clusters
        self.clusterer = clusterer
        self.affinity = affinity
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.eps = eps
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X):
        X = self._validate_data(X, accept_sparse=['csr', 'csc', 'coo'],
                                dtype=np.float64, ensure_min_samples=2)

        random_state = check_random_state(self.random_state)

        if self.clusterer is None:
            self.clusterer = KMeans(n_clusters=self.n_clusters,
                                    random_state=random_state)

        if self.eps is None:
            eps = 1e-5 / X.shape[0]
        else:
            eps = self.eps

        W = self._compute_norm_affinity_mat(X)

        v = random_state.random(size=X.shape[0])
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
        
        print('iters:', i)
        self.labels_ = self.clusterer.fit_predict(v.reshape(-1, 1))
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
            
        D_inv = np.reciprocal(A.sum(axis=1))
        W = D_inv.reshape(-1, 1) * A
        return W