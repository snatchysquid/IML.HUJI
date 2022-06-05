from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    # split data into cv folds by splitting the indices (without shuffling)
    total_indices = np.arange(X.shape[0])
    folds_idx = np.array_split(total_indices, cv)

    # init scores
    train_score = 0
    validation_score = 0

    for fold in folds_idx:
        # train indices are the indices that are not in the fold
        train_idx = np.setdiff1d(total_indices, fold)

        # split to train and cv
        train_X = X[train_idx]
        train_y = y[train_idx]
        cv_X = X[fold]
        cv_y = y[fold]

        # fit the estimator
        estimator.fit(train_X, train_y)

        # get the train and validation scores
        train_score += scoring(train_y, estimator.predict(train_X))
        validation_score += scoring(cv_y, estimator.predict(cv_X))

    # average scores
    train_score /= cv
    validation_score /= cv

    return train_score, validation_score


