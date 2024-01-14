import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy import linalg


class CSP(BaseEstimator, TransformerMixin):
    """
    CSP implementation based on MNE implementation

    https://github.com/mne-tools/mne-python/blob/f87be3000ce333ff9ccfddc45b47a2da7d92d69c/mne/decoding/csp.py
    """

    def __init__(self, n_components=4):
        self.n_components = n_components
        self.filters = None
        self.n_classes = None
        self.mean = None
        self.std = None

    def _decompose_covs(self, covs):
        """
         Return the eigenvalues and eigenvectors of a complex Hermitian ( conjugate symmetric )

        :param covs:
        :return:
        """
        from scipy import linalg
        n_classes = len(covs)
        if n_classes == 2:
            eigen_values, eigen_vectors = linalg.eigh(covs[0], covs.sum(0))
        else:
            raise Exception("Not Handled.")
        return eigen_vectors, eigen_values

    def calculate_cov_(self, X, y):
        _, n_channels, _ = X.shape
        covs = []

        for l in self.n_classes:
            lX = X[np.where(y == l)]
            lX = lX.transpose([1, 0, 2])
            lX = lX.reshape(n_channels, -1)
            covs.append(np.cov(lX))
        return np.asarray(covs)

    def calculate_eig_(self, covs):
        eigenvalues, eigenvectors = [], []
        for idx, cov in enumerate(covs):
            for iidx, compCov in enumerate(covs):
                if idx < iidx:
                    [eigVals, eigVects] = linalg.eig(cov, cov + compCov)
                    sorted_indices = np.argsort(np.abs(eigVals - 0.5))[::-1]
                    eigenvalues.append(eigVals[sorted_indices])
                    eigenvectors.append(eigVects[:, sorted_indices])
        return eigenvalues, eigenvectors

    def pick_filters(self, eigenvectors):
        filters = []
        for EigVects in eigenvectors:
            if filters == []:
                filters = EigVects[:, :self.n_components]
            else:
                filters = np.concatenate([filters, EigVects[:, :self.n_components]], axis=1)
        self.filters = filters.T

    def fit(self, X, y):
        self.n_classes = np.unique(y)

        if (len(self.n_classes) < 2):
            raise ValueError("n_classes must be >= 2")
        covs = self.calculate_cov_(X, y)
        eigenvalues, eigenvectors = self.calculate_eig_(covs)
        self.pick_filters(eigenvectors)
        X = np.asarray([np.dot(self.filters, epoch) for epoch in X])
        X = (X ** 2).mean(axis=2)

        # Standardize features
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)

    def transform(self, X):
        X = np.asarray([np.dot(self.filters, epoch) for epoch in X])
        X = (X ** 2).mean(axis=2)
        X -= self.mean
        X /= self.std
        return X

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
