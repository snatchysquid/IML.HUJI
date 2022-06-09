from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split as sklearn_tts

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    f = lambda x: (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)
    eps = np.random.normal(0, noise, n_samples)

    X = np.random.uniform(-1.2, 2, n_samples)
    y = f(X) + eps

    X_train, y_train, X_test, y_test = split_train_test(pd.Series(X), pd.Series(y), train_proportion=2 / 3)

    # make sure index is in order
    X_train.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

    # plot the data with different colors for training and testing
    fig = go.Figure(layout=go.Layout(title='Train & test data'))
    fig.add_trace(go.Scatter(x=X_train, y=f(X_train), mode='markers', name='Training')) \
        .add_trace(go.Scatter(x=X_test, y=f(X_test), mode='markers', name='Testing'))\
        .update_xaxes(title_text='x')\
        .update_yaxes(title_text='y')
    fig.show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    max_deg = 10
    train_scores = np.zeros(max_deg+1)
    validation_scores = np.zeros(max_deg+1)

    for deg in range(0, max_deg + 1):
        train_score, validation_score = cross_validate(PolynomialFitting(deg), X_train, y_train,
                                                       mean_square_error)

        train_scores[deg] = train_score
        validation_scores[deg] = validation_score

    # bar plot of train and validation scores
    fig = go.Figure(layout=go.Layout(title='Average Train & validation scores for different k values'))
    fig.add_trace(go.Bar(x=np.arange(0, max_deg + 1), y=train_scores, name='Training')) \
        .add_trace(go.Bar(x=np.arange(0, max_deg + 1), y=validation_scores, name='Validation'))\
        .update_xaxes(title_text='k')\
        .update_yaxes(title_text='score')
    fig.show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    best_k = np.argmin(validation_scores)

    polyfit = PolynomialFitting(best_k)
    polyfit.fit(X_train, y_train)

    cv_error = validation_scores[best_k]
    test_error = polyfit.loss(X_test, y_test)  # note: polyfit loss is in fact MSE, same loss used in cv

    print(f'Best k: {best_k}\nCV error: {cv_error}\nTest error: {test_error}')


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = sklearn_tts(X, y, train_size=n_samples)


    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    # we dont want to use 0 as it means no regularization and sklearn "discourages" this
    lambdas = np.linspace(0.001, 0.5, n_evaluations)

    lasso_train_scores = np.zeros(n_evaluations)
    lasso_validation_scores = np.zeros(n_evaluations)

    ridge_train_scores = np.zeros(n_evaluations)
    ridge_validation_scores = np.zeros(n_evaluations)

    for i, l in enumerate(lambdas):
        lasso = Lasso(l, max_iter=10000)
        lasso_train_score, lasso_validation_score = cross_validate(lasso, X_train, y_train, mean_square_error)
        lasso_train_scores[i] = lasso_train_score
        lasso_validation_scores[i] = lasso_validation_score

        ridge = RidgeRegression(l)
        ridge_train_score, ridge_validation_score = cross_validate(ridge, X_train, y_train, mean_square_error)
        ridge_train_scores[i] = ridge_train_score
        ridge_validation_scores[i] = ridge_validation_score

    fig = make_subplots(rows=2, cols=1, subplot_titles=('Lasso', 'Ridge'))
    fig.add_trace(go.Scatter(x=lambdas, y=lasso_train_scores, name='Training'), row=1, col=1) \
        .add_trace(go.Scatter(x=lambdas, y=lasso_validation_scores, name='Validation'), row=1, col=1)
    fig.add_trace(go.Scatter(x=lambdas, y=ridge_train_scores, name='Training'), row=2, col=1) \
        .add_trace(go.Scatter(x=lambdas, y=ridge_validation_scores, name='Validation'), row=2, col=1)
    fig.update_xaxes(title_text='lambda')\
        .update_yaxes(title_text='score')
    fig.update_layout(title='cv & train scores for different lambda values')
    fig.show()



    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    best_ridge = RidgeRegression(lambdas[np.argmin(ridge_validation_scores)])
    best_lasso = Lasso(lambdas[np.argmin(lasso_validation_scores)])
    least_squares = LinearRegression()

    best_ridge.fit(X_train, y_train)
    best_lasso.fit(X_train, y_train)
    least_squares.fit(X_train, y_train)

    print(f"Best ridge lambda: {lambdas[np.argmin(ridge_validation_scores)]}")
    print(f"Best lasso lambda: {lambdas[np.argmin(lasso_validation_scores)]}")

    print(f'Best Ridge model: {best_ridge.loss(X_test, y_test)}')
    print(f'Best Lasso model: {mean_square_error(best_lasso.predict(X_test), y_test)}')
    print(f'Least Squares model: {least_squares.loss(X_test, y_test)}')



if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    select_polynomial_degree(noise=0)
    select_polynomial_degree(n_samples=1500, noise=10)
    select_regularization_parameter()
