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
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        self.classes_ = np.unique(y)
        self.mu_ = np.zeros((self.classes_.size, X.shape[1]))
        self.vars_ = np.zeros((self.classes_.size, X.shape[1]))
        dataset = np.concatenate((X, y.reshape(-1, 1)), axis=1)

        self.pi_ = np.array([np.sum(dataset[:, -1] == _class) for _class in self.classes_]) / len(dataset)

        for i, _class in enumerate(self.classes_):
            slice_dataset = dataset[dataset[:, -1] == _class][:, :-1]

            # set ddof in case we have a single sample from that class
            ddof = 0 if slice_dataset.shape[0] == 1 else 1

            self.mu_[i] = np.mean(slice_dataset, axis=0)
            self.vars_[i] = np.var(slice_dataset, axis=0, ddof=ddof)

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
        return self.classes_[np.nanargmax(likelihoods, axis=1)]



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

        likelihoods = np.zeros((X.shape[0], self.classes_.size))
        sigma_prod = 1 / np.prod(np.sqrt(self.vars_), axis=1)

        for i in range(self.classes_.shape[0]):
            exp_part = np.exp(-0.5 * np.sum((X - self.mu_[i]) ** 2 / self.vars_[i], axis=1))
            likelihoods[:, i] = sigma_prod[i] * exp_part / ((2 * np.pi) ** (X.shape[1] / 2))

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
<<<<<<< HEAD
        return misclassification_error(y, self.predict(X))
||||||| bfc7971
        raise NotImplementedError()
=======
        from ...metrics import misclassification_error
        raise NotImplementedError()
>>>>>>> c87be5d7872d40b4409d315bf2d2360bc8a3d675
