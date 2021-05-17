import numpy as np
from sklearn.cluster import KMeans
import pandas as pd

class PIC:
    def __init__(self, kmeans, eps=None, max_iter=1000):
        self.kmeans = kmeans
        self.eps = eps
        self.max_iter = max_iter

    def fit_predict(self, X):
        X = X.values if isinstance(X, pd.DataFrame) else X

        eps = 1e-5 / X.shape[0] if self.eps is None else self.eps

        # sim
        sim = cosine_sim

        W = self._compute_norm_affinity(X, sim)

        v = np.random.random(size=X.shape[0])
        v_old = v + 100 * eps
        delta_old = v - v_old
        
        for i in range(self.max_iter):
            wv = W @ v
            v = wv / np.abs(wv).sum()
            delta = v - v_old

            if np.all(np.abs(delta - delta_old) < eps):
                print('break at:', i)
                break

            delta_old = delta
            v_old = v
        
        print('iters:', i)
        self.v_ = v
        self.W_ = W

        # v2 = np.vstack([v, list(range(len(v)))]).T
        return self.kmeans.fit_predict(v.reshape(-1, 1))

    def predict(self, X):
        return self.kmeans.predict(X)

    def _compute_norm_affinity(self, X, sim):
        A = np.empty((X.shape[0], X.shape[0]))

        for i in range(X.shape[0]):
            A[i, i:] = sim(X[i][None, ...], X[i:])
            A[i:, i] = A[i, i:]
            
        # A =  A / A.sum(axis=0)
        # W = np.linalg.inv(np.diag(A.sum(axis=1))) @ A
        D_inv = np.reciprocal(A.sum(axis=1))
        W = D_inv.reshape(-1, 1) * A
        return W


def cosine_sim(x, y):
    x_norm = np.linalg.norm(x, axis=1)
    y_norm = np.linalg.norm(y, axis=1)
    return np.dot(x, y.T) / (x_norm * y_norm)


