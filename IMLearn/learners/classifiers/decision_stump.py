from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from itertools import product
from ...metrics.loss_functions import misclassification_error


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """
    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        feature_err = np.empty((X.shape[1], 2), dtype=np.float64)
        thresholds = np.empty((X.shape[1], 2), dtype=np.float64)

        for j in range(X.shape[1]):
            thresholds[j, 0], feature_err[j, 0] = self._find_threshold(X[:, j], y, 1)
            thresholds[j, 1], feature_err[j, 1] = self._find_threshold(X[:, j], y, -1)  # thresholds[j, 0], 1 - feature_err[j, 0]

        # get feature index and sign with lowest error
        self.j_, self.sign_ = np.unravel_index(np.argmin(feature_err), feature_err.shape)
        self.threshold_ = thresholds[self.j_, self.sign_]

        # note that now sign is either 0 or 1 and not -1 or 1
        # this is because the sign is also used as the index here in thresholds and feature_err
        # so we need to change the sign
        self.sign_ = 1 if self.sign_ == 0 else -1

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        j_feature = X[:, self.j_]

        responses = np.ones(X.shape[0], dtype=np.int8)
        responses[j_feature <= self.threshold_] = self.sign_
        responses[j_feature > self.threshold_] = -self.sign_

        return responses


    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        # # sort values and labels by values
        sorted_indices = np.argsort(values)
        sorted_values, sorted_labels = values[sorted_indices], labels[sorted_indices]
        abs_labels = np.abs(sorted_labels)

        leftmost_threshold_correct = np.sum(abs_labels * (np.sign(sorted_labels) == sign))  # num of correct labels for leftmost threshold

        # we use cumulative sum because each time we go to the next threshold,
        # only one value is effect - the next value in the cumsum
        # if label*sign = 1 we are correct, which means we were wrong before so we add 1
        # if label*sign = -1 we are wrong, which means we were correct before so we subtract 1
        # this is exactly what cumsum does here
        # the base for that is "leftmost_threshold_correct", so we add the cumsum to it.
        threshold_corrects = leftmost_threshold_correct - np.cumsum(sorted_labels * sign)

        # get maximal gain
        max_correct = np.argmin(threshold_corrects)

        # compare max_correct and leftmost_threshold_correct
        if threshold_corrects[max_correct] >= leftmost_threshold_correct:
            # return very small number and the loss (1 - corrects_ratio)
            return np.NINF, leftmost_threshold_correct / len(sorted_labels)
        elif max_correct == threshold_corrects.shape[0] or sorted_values[max_correct] == sorted_values[-1]:  # if max_correct is the last index (rightmost border), return inf
            # return very large number and the loss (1 - corrects_ratio)
            return np.inf, sorted_values[max_correct] / len(sorted_labels)

        return sorted_values[max_correct], threshold_corrects[max_correct] / len(sorted_labels)


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
        return misclassification_error(y, self._predict(X))
