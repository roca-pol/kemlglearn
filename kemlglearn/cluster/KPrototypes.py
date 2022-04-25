
"""K-prototypes clustering"""

# Author: Nico de Vos <njdevos@gmail.com>
# License: MIT

from collections import defaultdict
import numpy as np
from .KModes import KModes


def euclidean_dissim(a, b):
    """Euclidean distance dissimilarity function"""
    return np.sum((a - b) ** 2, axis=1)


def move_point_num(point, ipoint, to_clust, from_clust,
                   cl_attr_sum, membership):
    """Move point between clusters, numerical attributes."""
    membership[to_clust, ipoint] = 1
    membership[from_clust, ipoint] = 0
    # Update sum of attributes in cluster.
    for iattr, curattr in enumerate(point):
        cl_attr_sum[to_clust][iattr] += curattr
        cl_attr_sum[from_clust][iattr] -= curattr
    return cl_attr_sum, membership


def _labels_cost(Xnum, Xcat, centroids, gamma):
    """Calculate labels and cost function given a matrix of points and
    a list of centroids for the k-prototypes algorithm.
    """

    npoints = Xnum.shape[0]
    cost = 0.
    labels = np.empty(npoints, dtype='int64')
    for ipoint in range(npoints):
        # Numerical cost = sum of Euclidean distances
        num_costs = euclidean_dissim(centroids[0], Xnum[ipoint])
        cat_costs = KModes.matching_dissim(centroids[1], Xcat[ipoint])
        # Gamma relates the categorical cost to the numerical cost.
        tot_costs = num_costs + gamma * cat_costs
        clust = np.argmin(tot_costs)
        labels[ipoint] = clust
        cost += tot_costs[clust]

    return labels, cost


def _k_prototypes_iter(Xnum, Xcat, centroids, cl_attr_sum, cl_attr_freq,
                       membership, gamma):
    """Single iteration of the k-prototypes algorithm"""
    moves = 0
    for ipoint in range(Xnum.shape[0]):
        clust = np.argmin(
            euclidean_dissim(centroids[0], Xnum[ipoint]) +
            gamma * KModes.matching_dissim(centroids[1], Xcat[ipoint]))
        if membership[clust, ipoint]:
            # Point is already in its right place.
            continue

        # Move point, and update old/new cluster frequencies and centroids.
        moves += 1
        old_clust = np.argwhere(membership[:, ipoint])[0][0]

        cl_attr_sum, membership = move_point_num(
            Xnum[ipoint], ipoint, clust, old_clust, cl_attr_sum,
            membership)
        cl_attr_freq, membership = KModes.move_point_cat(
            Xcat[ipoint], ipoint, clust, old_clust, cl_attr_freq,
            membership)

        # Update new and old centroids by choosing mean for numerical
        # and mode for categorical attributes.
        for iattr in range(len(Xnum[ipoint])):
            for curc in (clust, old_clust):
                if sum(membership[curc, :]):
                    centroids[0][curc, iattr] = \
                        cl_attr_sum[curc, iattr] / sum(membership[curc, :])
                else:
                    centroids[0][curc, iattr] = 0.
        for iattr in range(len(Xcat[ipoint])):
            for curc in (clust, old_clust):
                centroids[1][curc, iattr] = \
                    KModes.get_max_value_key(cl_attr_freq[curc][iattr])

        # In case of an empty cluster, reinitialize with a random point
        # from largest cluster.
        if sum(membership[old_clust, :]) == 0:
            from_clust = membership.sum(axis=1).argmax()
            choices = \
                [ii for ii, ch in enumerate(membership[from_clust, :]) if ch]
            rindx = np.random.choice(choices)

            cl_attr_freq, membership = move_point_num(
                Xnum[rindx], rindx, old_clust, from_clust, cl_attr_sum,
                membership)
            cl_attr_freq, membership = KModes.move_point_cat(
                Xcat[rindx], rindx, old_clust, from_clust, cl_attr_freq,
                membership)

    return centroids, moves


def k_prototypes(X, n_clusters, gamma, init, n_init, max_iter, verbose):
    """k-prototypes algorithm"""

    assert len(X) == 2, "X should be a list of Xnum and Xcat arrays"
    # List where [0] = numerical part of centroid and
    # [1] = categorical part. Same for centroids.
    Xnum, Xcat = X
    # Convert to numpy arrays, if needed.
    Xnum = np.asanyarray(Xnum)
    Xcat = np.asanyarray(Xcat)
    nnumpoints, nnumattrs = Xnum.shape
    ncatpoints, ncatattrs = Xcat.shape
    assert nnumpoints == ncatpoints,\
        "Uneven number of numerical and categorical points"
    npoints = nnumpoints
    assert n_clusters < npoints, "More clusters than data points?"

    # Estimate a good value for gamma, which determines the weighing of
    # categorical values in clusters (see Huang [1997]).
    if gamma is None:
        gamma = 0.5 * Xnum.std()

    all_centroids = []
    all_labels = []
    all_costs = []
    for init_no in range(n_init):

        # For numerical part of initialization, we don't have a guarantee
        # that there is not an empty cluster, so we need to retry until
        # there is none.
        while True:
            # _____ INIT _____
            if verbose:
                print("Init: initializing centroids")
            if init == 'Huang':
                centroids = KModes.init_huang(Xcat, n_clusters)
            elif init == 'Cao':
                centroids = KModes.init_cao(Xcat, n_clusters)
            elif init == 'random':
                seeds = np.random.choice(range(npoints), n_clusters)
                centroids = Xcat[seeds]
            elif hasattr(init, '__array__'):
                centroids = init
            else:
                raise NotImplementedError

            # Numerical is initialized by drawing from normal distribution,
            # categorical following the k-modes methods.
            meanX = np.mean(Xnum, axis=0)
            stdX = np.std(Xnum, axis=0)
            centroids = [meanX + np.random.randn(n_clusters, nnumattrs) * stdX,
                         centroids]

            if verbose:
                print("Init: initializing clusters")
            membership = np.zeros((n_clusters, npoints), dtype='int64')
            # Keep track of the sum of attribute values per cluster so that we
            # can do k-means on the numerical attributes.
            cl_attr_sum = np.zeros((n_clusters, nnumattrs), dtype='float')
            # cl_attr_freq is a list of lists with dictionaries that contain
            # the frequencies of values per cluster and attribute.
            cl_attr_freq = [[defaultdict(int) for _ in range(ncatattrs)]
                            for _ in range(n_clusters)]
            for ipoint in range(npoints):
                # Initial assignment to clusters
                clust = np.argmin(
                    euclidean_dissim(centroids[0], Xnum[ipoint]) +
                    gamma * KModes.matching_dissim(centroids[1], Xcat[ipoint]))
                membership[clust, ipoint] = 1
                # Count attribute values per cluster.
                for iattr, curattr in enumerate(Xnum[ipoint]):
                    cl_attr_sum[clust, iattr] += curattr
                for iattr, curattr in enumerate(Xcat[ipoint]):
                    cl_attr_freq[clust][iattr][curattr] += 1

            # If no empty clusters, then consider initialization finalized.
            if membership.sum(axis=1).min() > 0:
                break

        # Perform an initial centroid update.
        for ik in range(n_clusters):
            for iattr in range(nnumattrs):
                centroids[0][ik, iattr] =  \
                    cl_attr_sum[ik, iattr] / sum(membership[ik, :])
            for iattr in range(ncatattrs):
                centroids[1][ik, iattr] = \
                    KModes.get_max_value_key(cl_attr_freq[ik][iattr])

        # _____ ITERATION _____
        if verbose:
            print("Starting iterations...")
        itr = 0
        converged = False
        cost = np.Inf
        while itr <= max_iter and not converged:
            itr += 1
            centroids, moves = _k_prototypes_iter(
                Xnum, Xcat, centroids, cl_attr_sum, cl_attr_freq,
                membership, gamma)

            # All points seen in this iteration
            labels, ncost = \
                _labels_cost(Xnum, Xcat, centroids, gamma)
            converged = (moves == 0) or (ncost >= cost)
            cost = ncost
            if verbose:
                print(
                    f"Run: {init_no + 1}, iteration: {itr}/{max_iter}, moves: {moves}, ncost: {ncost}"
                )


        # Store results of current run.
        all_centroids.append(centroids)
        all_labels.append(labels)
        all_costs.append(cost)

    best = np.argmin(all_costs)
    if n_init > 1 and verbose:
        print(f"Best run was number {best + 1}")

    # Note: return gamma in case it was automatically determined.
    return all_centroids[best], all_labels[best], all_costs[best], gamma


class KPrototypes(KModes):
    """k-protoypes clustering algorithm for mixed numerical/categorical data.

    Parameters
    -----------
    n_clusters : int, optional, default: 8
        The number of clusters to form as well as the number of
        centroids to generate.

    gamma : float, default: None
        Weighing factor that determines relative importance of numerical vs.
        categorical attributes (see discussion in Huang [1997]). By default,
        automatically calculated from data.

    max_iter : int, default: 300
        Maximum number of iterations of the k-modes algorithm for a
        single run.

    n_init : int, default: 10
        Number of time the k-modes algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of cost.

    init : {'Huang', 'Cao', 'random' or an ndarray}
        Method for initialization:
        'Huang': Method in Huang [1997, 1998]
        'Cao': Method in Cao et al. [2009]
        'random': choose k observations (rows) at random from data for
        the initial centroids.
        If an ndarray is passed, it should be of shape (n_clusters, n_features)
        and gives the initial centroids.

    verbose : boolean, optional
        Verbosity mode.

    Attributes
    ----------
    cluster_centroids_ : array, [n_clusters, n_features]
        Categories of cluster centroids

    labels_ :
        Labels of each point

    cost_ : float
        Clustering cost, defined as the sum distance of all points to
        their respective cluster centroids.

    Notes
    -----
    See:
    Huang, Z.: Extensions to the k-modes algorithm for clustering large
    data sets with categorical values, Data Mining and Knowledge
    Discovery 2(3), 1998.

    """

    def __init__(self, n_clusters=8, gamma=None, init='Huang', n_init=10,
                 max_iter=100, verbose=0):

        super(KPrototypes, self).__init__(n_clusters, init, n_init, max_iter,
                                          verbose)

        self.gamma = gamma

    def fit(self, X):
        """Compute k-prototypes clustering.

        Parameters
        ----------
        X : list of array-like, shape=[[n_num_samples, n_features],
                                       [n_cat_samples, n_features]]
        """

        # If self.gamma is None, gamma will be automatically determined from
        # the data. The function below returns its value.
        self.cluster_centroids_, self.labels_, self.cost_, self.gamma = \
            k_prototypes(X, self.n_clusters, self.gamma, self.init,
                         self.n_init, self.max_iter, self.verbose)
        return self

    def predict(self, X):
        """Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : list of array-like, shape=[[n_num_samples, n_features],
                                       [n_cat_samples, n_features]]

        Returns
        -------
        labels : array, shape [n_samples,]
            Index of the cluster each sample belongs to.
        """
        assert hasattr(self, 'cluster_centroids_'), "Model not yet fitted."
        return _labels_cost(X[0], X[1], self.cluster_centroids_,
                            self.gamma)[0]
