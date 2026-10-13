"""Kernel Naive Bayes: per-feature class-conditional densities estimated with KDE."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KernelDensity


class KernelNaiveBayes(BaseEstimator, ClassifierMixin):
    def __init__(self, bandwidth=0.1, kernel="gaussian"):
        self.bandwidth = bandwidth
        self.kernel = kernel

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.log_priors_ = np.log(np.array([(y == c).mean() for c in self.classes_]))
        self.kdes_ = []
        for c in self.classes_:
            Xc = X[y == c]
            self.kdes_.append([
                KernelDensity(kernel=self.kernel, bandwidth=self.bandwidth, rtol=1e-3).fit(Xc[:, [j]])
                for j in range(X.shape[1])
            ])
        return self

    def _joint_log_likelihood(self, X):
        X = np.asarray(X, dtype=float)
        jll = np.zeros((X.shape[0], len(self.classes_)))
        for k, kde_list in enumerate(self.kdes_):
            log_lik = sum(kde.score_samples(X[:, [j]]) for j, kde in enumerate(kde_list))
            jll[:, k] = self.log_priors_[k] + log_lik
        return jll

    def predict(self, X):
        return self.classes_[np.argmax(self._joint_log_likelihood(X), axis=1)]

    def predict_proba(self, X):
        jll = self._joint_log_likelihood(X)
        jll -= jll.max(axis=1, keepdims=True)
        p = np.exp(jll)
        return p / p.sum(axis=1, keepdims=True)
