from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from ...metrics import misclassification_error

class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """
    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_ = np.unique(y).astype(int)
        self.mu_ = np.zeros((self.classes_.size, X.shape[1]))
        self.vars_ = np.zeros((self.classes_.size, X.shape[1]))
        dataset = np.concatenate((X, y.reshape(-1, 1)), axis=1)

        self.pi_ = np.array([np.sum(dataset[:, -1] == _class) for _class in self.classes_]) / len(dataset)

        for _class in self.classes_:
            self.mu_[_class] = np.mean(dataset[dataset[:, -1] == _class][:, :-1], axis=0)
            self.vars_[_class] = np.var(dataset[dataset[:, -1] == _class][:, :-1], axis=0)

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
        likelihoods = self.likelihood(X)
        return np.nanargmax(likelihoods, axis=1)



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

        likelihoods = np.zeros((X.shape[0], self.classes_.size))
        sigma_prod = 1 / np.prod(np.sqrt(self.vars_), axis=1)

        for _class in self.classes_:
            exp_part = np.exp(-0.5 * np.sum((X - self.mu_[_class]) ** 2 / self.vars_[_class], axis=1))
            likelihoods[:, _class] = sigma_prod[_class] * exp_part / ((2 * np.pi) ** (X.shape[1] / 2))

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
