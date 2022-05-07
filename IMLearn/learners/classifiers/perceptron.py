from __future__ import annotations
from typing import Callable
from typing import NoReturn
from ...base import BaseEstimator
from ...metrics import misclassification_error
import numpy as np


def default_callback(fit: Perceptron, x: np.ndarray, y: int):
    pass


class Perceptron(BaseEstimator):
    """
    Perceptron half-space classifier

    Finds a separating hyperplane for given linearly separable data.

    Attributes
    ----------
    include_intercept: bool, default = True
        Should fitted model include an intercept or not

    max_iter_: int, default = 1000
        Maximum number of passes over training data

    coefs_: ndarray of shape (n_features,) or (n_features+1,)
        Coefficients vector fitted by Perceptron algorithm. To be set in
        `Perceptron.fit` function.

    callback_: Callable[[Perceptron, np.ndarray, int], None]
            A callable to be called after each update of the model while fitting to given data
            Callable function should receive as input a Perceptron instance, current sample and current response
    """
    def __init__(self,
                 include_intercept: bool = True,
                 max_iter: int = 1000,
                 callback: Callable[[Perceptron, np.ndarray, int], None] = default_callback):
        """
        Instantiate a Perceptron classifier

        Parameters
        ----------
        include_intercept: bool, default=True
            Should fitted model include an intercept or not

        max_iter: int, default = 1000
            Maximum number of passes over training data

        callback: Callable[[Perceptron, np.ndarray, int], None]
            A callable to be called after each update of the model while fitting to given data
            Callable function should receive as input a Perceptron instance, current sample and current response
        """
        super().__init__()
        self.include_intercept_ = include_intercept
        self.max_iter_ = max_iter
        self.callback_ = callback
        self.coefs_ = None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit a halfspace to to given samples. Iterate over given data as long as there exists a sample misclassified
        or that did not reach `self.max_iter_`

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Notes
        -----
        Fits model with or without an intercept depending on value of `self.fit_intercept_`
        """
        # choose whether to include intercept or not (add 1 to X)
        self.fitted_ = True

        if self.include_intercept_:
            X = np.c_[np.ones(X.shape[0]), X]

        self.coefs_ = np.zeros(X.shape[1])

        for _ in range(self.max_iter_):
            did_change = False  # var for checking if we converged

            # gives indices where value is False (where a mistake was made)
            mistake_indices = np.where(np.dot(X, self.coefs_) * y <= 0)[0]

            if len(mistake_indices):
                # update coefficients
                first_mistake = mistake_indices[0]
                self.coefs_ += y[first_mistake] * X[first_mistake]
                did_change = True

                self.callback_(self, X[first_mistake], y[first_mistake])

            # check if we converged early so we can exit loop
            if not did_change:
                break


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
        if self.include_intercept_:
            X = np.c_[np.ones(X.shape[0]), X]

        prediction = (np.dot(X, self.coefs_) > 0).astype(int)
        prediction[prediction == 0] = -1
        return prediction

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
        return misclassification_error(y, self._predict(X))
||||||| bfc7971
        raise NotImplementedError()
=======
        from ...metrics import misclassification_error
        raise NotImplementedError()
>>>>>>> c87be5d7872d40b4409d315bf2d2360bc8a3d675
