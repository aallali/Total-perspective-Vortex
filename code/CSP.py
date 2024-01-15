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

    def calculate_cov_(self, X, y):
        """
        Calculate the covariance matrices for each class.

        Parameters:
        - X: ndarray, shape (n_epochs, n_channels, n_times)
            The EEG data.
        - y: array, shape (n_epochs,)
            The labels for each epoch.

        Returns:
        - covs: ndarray, shape (n_classes, n_channels, n_channels)
            List of covariance matrices for each class.
        """

        _, n_channels, _ = X.shape
        covs = []

        # Iterate over each class
        for l in self.n_classes:
            # Select epochs corresponding to the current class
            lX = X[np.where(y == l)]
            # Transpose to have shape (n_channels, n_epochs, n_times)
            lX = lX.transpose([1, 0, 2])
            # Reshape to (n_channels, -1)
            lX = lX.reshape(n_channels, -1)
            # Calculate covariance matrix for the class
            # The covariance matrix is a square matrix \
            #   where each element represents the covariance \
            #   between two corresponding channels (features) in the input data.
            covs.append(np.cov(lX)) # type: 'numpy.ndarray'

        return np.asarray(covs)

    def calculate_eig_(self, covs):
        """
        Calculate eigenvalues and eigenvectors for pairwise combinations of covariance matrices.

        Parameters:
        -----------
        covs : list of 2D arrays
            List of covariance matrices for different classes.

        Returns:
        --------
        tuple
            Tuple containing lists of eigenvalues and eigenvectors.

        """
        eigenvalues, eigenvectors = [], []

        # Iterate over each covariance matrix
        for idx, cov in enumerate(covs):
            # Iterate over remaining covariance matrices to create pairwise combinations
            for iidx, compCov in enumerate(covs):
                if idx < iidx:  # Consider each pair only once
                    # Solve the generalized eigenvalue problem
                    eigVals, eigVects = linalg.eig(cov, cov + compCov)
                    # Sort eigenvalues in descending order
                    sorted_indices = np.argsort(np.abs(eigVals - 0.5))[::-1]
                    # Store sorted eigenvalues and corresponding eigenvectors
                    eigenvalues.append(eigVals[sorted_indices])
                    eigenvectors.append(eigVects[:, sorted_indices])

        return eigenvalues, eigenvectors

    def pick_filters(self, eigenvectors):
        """
        Select CSP filters based on the sorted eigenvectors.

        Parameters:
        -----------
        eigenvectors : list of 2D arrays
            List of eigenvectors corresponding to each pairwise combination of covariance matrices.

        Returns:
        --------
        None
            Updates the `filters` attribute with the selected CSP filters.

        """
        filters = []

        # Iterate over each set of eigenvectors
        for EigVects in eigenvectors:
            # If filters is empty, directly assign the first set of eigenvectors
            if filters == []:
                filters = EigVects[:, :self.n_components]
            else:
                # Concatenate the current set of eigenvectors to the existing filters
                filters = np.concatenate([filters, EigVects[:, :self.n_components]], axis=1)

        # Transpose the filters matrix and store it in the `filters` attribute
        self.filters = filters.T

    def fit(self, X, y):
        self.n_classes = np.unique(y)

        if len(self.n_classes) < 2:
            raise ValueError("n_classes must be >= 2")

        # Calculate the covariance matrices for each class
        covs = self.calculate_cov_(X, y)

        # Calculate the eigenvalues and eigenvectors for the covariances
        eigenvalues, eigenvectors = self.calculate_eig_(covs)

        # Pick the CSP filters based on eigenvalues and eigenvectors
        self.pick_filters(eigenvectors)

        # Transform the input data using the selected CSP filters
        X = np.asarray([np.dot(self.filters, epoch) for epoch in X])
        X = (X ** 2).mean(axis=2)

        # Standardize features
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)

    def transform(self, X):
        # Transform the input data using the selected CSP filters
        X = np.asarray([np.dot(self.filters, epoch) for epoch in X])

        # Square and average along the time axis
        X = (X ** 2).mean(axis=2)

        # Standardize features
        X -= self.mean
        X /= self.std
        """
        example:
            X = [[2, 4, 6],
                 [1, 3, 5],
                 [3, 5, 7]]
            mean = [2, 4, 6]
            X -= mean
                
            Result:
            X = [[0, 0, 0],
                 [-1, -1, -1],
                 [1, 1, 1]]
            std = [0.8165, 0.8165, 0.8165]
            X /= std
            
            Result:
            X = [[0, 0, 0],
                 [-1.2247, -1.2247, -1.2247],
                 [1.2247, 1.2247, 1.2247]]
        """
        return X

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
