from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv
from ...metrics import misclassification_error


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """
    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        self.classes_ = np.unique(y)

        dataset = np.concatenate((X, y.reshape(-1, 1)), axis=1)
        self.mu_ = np.array([np.mean(dataset[dataset[:, -1] == label][:, :-1], axis=0) for label in self.classes_])

        self.cov_ = np.zeros((X.shape[1], X.shape[1]))


        for i_sample in range(X.shape[0]):
            # normalized = X[i_sample] - self.mu_[int(y[i_sample])]
            normalized = X[i_sample] - self.mu_[np.where(self.classes_ == y[i_sample])[0]]
            self.cov_ += np.outer(normalized, normalized)

        self.cov_ /= X.shape[0] - self.classes_.shape[0]

        self._cov_inv = inv(self.cov_)

        self.pi_ = np.array([np.sum(dataset[:, -1] == _class) for _class in self.classes_]) / len(dataset)


    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        a_classes = self._cov_inv @ self.mu_.T
        b_classes = np.log(self.pi_) - 0.5 * np.sum(self.mu_.T * (self._cov_inv @ self.mu_.T), axis=0)

        results = X @ a_classes + b_classes
        return self.classes_[np.nanargmax(results, axis=1)]

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        likelihoods = np.zeros((X.shape[0], len(self.classes_)))

        for i in range(self.classes_.shape[0]):
            class_samples = X - self.mu_[i]
            likelihoods[:, i] = np.exp(-0.5 * np.sum(class_samples.T * (self._cov_inv @ class_samples.T), axis=0))

        likelihoods /= np.sqrt(np.linalg.det(self.cov_) * (2 * np.pi) ** X.shape[1])
        likelihoods *= self.pi_

        return likelihoods

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        return misclassification_error(y, self.predict(X))
