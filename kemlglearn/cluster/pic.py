import numpy as np
from sklearn.cluster import KMeans
from sklearn.base import ClusterMixin, BaseEstimator
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils.validation import check_random_state
from scipy.sparse import spdiags, issparse


class PIC(ClusterMixin, BaseEstimator):
    """Apply clustering to a projection of the normalized affinity matrix.

    This algorithm is similar to spectral clustering in that both can be
    used to seek the normalized graph cuts. This method, however, does not
    have to compute the eigenvalue decomposition of the affinity matrix
    because it finds an approximation of the top eigenvector by power
    iteration, which is faster.

    When calling ``fit``, an affinity matrix is constructed using a
    kernel function such the Gaussian (aka RBF) kernel of the euclidean
    distanced ``d(X, X)``::

            np.exp(-gamma * d(X,X) ** 2)

    Alternatively, using ``precomputed``, a user-provided affinity
    matrix can be used.

    Parameters
    ----------
    n_clusters : integer
        The number of partitions to seek in the projection subspace
        using KMeans.

    affinity : string or callable, default 'cosine'
        How to construct the affinity matrix.
         - 'rbf' : construct the affinity matrix using a radial basis function
           (RBF) kernel.
         - 'precomputed' : interpret ``X`` as a precomputed affinity matrix.
         - one of the kernels supported by
           :func:`~sklearn.metrics.pairwise_kernels`.

    init : 'random' or 'degree', default: 'random'
        How to generate the initial vector v. 'random' means randomly,
        and 'degree' means to use the normalized diagonal of the degree
        matrix.

    max_iter : int, optional, default: 50
        Maximum number of iteration to perform power iteration.

    eps : float or 'auto', optional, default: 'auto'
        Stopping criterion for the convergence of the power iteration
        based on the stability of the approximated eigenvector. When
        set to 'auto' it will be calculated when ``fit`` is called with
        this formula: eps = 1e-5 / n where n is the number of instances.

    random_state : int, RandomState instance, default=None
        A pseudo random number generator used for the initialization of the
        vector of the power iteration method and by the KMeans initialization.
        Use an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.

    n_init : int, optional, default: 10
        Number of time the KMeans algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of inertia.

    gamma : float, default=1.0
        Kernel coefficient for rbf, poly, sigmoid, laplacian and chi2 kernels.

        Only kernels that produce similarity scores (non-negative values that
        increase with similarity) should be used. This property is not checked
        by the clustering algorithm.

    degree : float, default=3
        Degree of the polynomial kernel. Ignored by other kernels.

    coef0 : float, default=1
        Zero coefficient for polynomial and sigmoid kernels.
        Ignored by other kernels.

    kernel_params : dictionary of string to any, optional
        Parameters (keyword arguments) and values for kernel passed as
        callable object. Ignored by other kernels.

    Attributes
    ----------
    affinity_matrix_ : array-like, shape (n_samples, n_samples)
        Affinity matrix used for clustering. Available only if after calling

    v_ : array, shape (n_samples,)
        The result from power iteration, a one dimensional vector where
        data points have been embedded to.
        ``fit``.

    labels_ : array, shape (n_samples,)
        Labels of each point.

    n_iter_ : int
        Number of iterations that the algorithm was run.

    References
    ----------

    - Power Iteration Clustering, 2010
      Frank Lin, William W. Cohen
      https://dl.acm.org/doi/10.5555/3104322.3104406
    """
    def __init__(self, n_clusters=2, affinity='cosine', init='random',
                 gamma=1.0, degree=3, coef0=1, kernel_params=None, 
                 eps='auto', max_iter=50, n_init=1, random_state=None):
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

    def fit(self, X, y=None):
        """Perform power iteration clustering from features, or affinity matrix.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features), or \
            array-like, shape (n_samples, n_samples)
            Training instances to cluster, or similarities / affinities between
            instances if ``affinity='precomputed'``. If a sparse matrix is
            provided in a format other than ``csr_matrix``, ``csc_matrix``,
            or ``coo_matrix``, it will be converted into a sparse
            ``csr_matrix``.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self

        """
        X = self._validate_data(X, accept_sparse=['csr', 'csc', 'coo'],
                                dtype=np.float64, ensure_min_samples=2)

        eps = 1e-5 / X.shape[0] if self.eps == 'auto' else self.eps
        random_state = check_random_state(self.random_state)
        if self.init == 'degree':
            W, v = self._compute_norm_affinity_mat(X, return_degree=True)
        else:
            W = self._compute_norm_affinity_mat(X)
            v = random_state.random(size=W.shape[0])

        delta = 1e3  # force 2 iterations at minimum

        # perform power iteartion
        for i in range(self.max_iter):
            v_old = v
            delta_old = delta

            u = W @ v
            v = u / np.abs(u).sum()
            delta = v - v_old

            if np.all(np.abs(delta - delta_old) < eps):
                break

        # cluster the one dimensional embedded space
        kmeans = KMeans(n_clusters=self.n_clusters,
                        n_init=self.n_init,
                        random_state=random_state)
        self.labels_ = kmeans.fit_predict(v.reshape(-1, 1))
        self.v_ = v
        self.affinity_matrix_ = W
        self.n_iter_ = i + 1
        return self

    def _compute_norm_affinity_mat(self, X, return_degree=False):
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
            np.fill_diagonal(A, 0.)
        
        # normalize
        D = A.sum(axis=1)
        D_inv = np.reciprocal(D)
        if issparse(A):
            D_inv = spdiags(D_inv.reshape(1, -1), 0, len(D_inv), len(D_inv))
            W = D_inv @ A
        else:
            # same as multiplying D_inv @ A
            # but we can broadcast which is faster
            W = D_inv.reshape(-1, 1) * A
            
        if return_degree:
            v0 = D / D.sum()
            return W, v0
        else:
            return W