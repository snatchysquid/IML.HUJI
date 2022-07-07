import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.metrics.loss_functions import misclassification_error

from sklearn.metrics import roc_curve, auc
import plotly.graph_objects as go


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """
    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines", marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    values = []
    weights = []

    def callback(solver, weight, val, grad, t, eta, delta):
        values.append(val)
        weights.append(weight)

    return callback, values, weights


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):

    # create fig holding the norm of the weights for each learning rate at each step
    norm_fig = go.Figure()


    for eta in etas:
        lr = FixedLR(eta)

        l1 = L1(weights=init)
        l2 = L2(weights=init)

        l1_callback, l1_values, l1_weights = get_gd_state_recorder_callback()
        gd = GradientDescent(learning_rate=lr, callback=l1_callback, max_iter=3000)
        gd.fit(f=l1, X=None, y=None)

        l2_callback, l2_values, l2_weights = get_gd_state_recorder_callback()
        gd = GradientDescent(learning_rate=lr, callback=l2_callback, max_iter=3000)
        gd.fit(f=l2, X=None, y=None)

        # l1_weights is a list of the form [array(...), array(...), ...]
        # we want to turn it to np.array of shape (n_iterations, n_features)
        l1_weights = np.array([w for w in l1_weights])
        l2_weights = np.array([w for w in l2_weights])

        # add the norm of the weights to the figure
        norm_fig.add_trace(go.Scatter(x=np.arange(len(l1_weights)), y=np.linalg.norm(l1_weights, axis=1),
                                        name=f"l1 eta={eta}"))
        norm_fig.add_trace(go.Scatter(x=np.arange(len(l2_weights)), y=np.linalg.norm(l2_weights, axis=1),
                                        name=f"l2 eta={eta}"))

        # plot the results
        fig = plot_descent_path(L1, l1_weights, title=f"L1, eta={eta}")
        fig.update_xaxes(title="x").update_yaxes(title="y")
        fig.show()

        fig = plot_descent_path(L2, l2_weights, title=f"L2, eta={eta}")
        fig.update_xaxes(title="x").update_yaxes(title="y")
        fig.show()

        print(f"min L2, eta={eta}:")
        print(np.min(l2_values))
        print(f"min L1, eta={eta}:")
        print(np.min(l1_values))

    norm_fig.update_xaxes(title="Iteration").update_yaxes(title="Norm of Weights").update_layout(title="Norm of Weights")
    norm_fig.show()





def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    # create fig holding the norm of the weights for each learning rate at each step
    norm_fig = go.Figure()



    for gamma in gammas:
        lr = ExponentialLR(base_lr=eta, decay_rate=gamma)

        l1 = L1(weights=init)

        l1_callback, l1_values, l1_weights = get_gd_state_recorder_callback()
        gd = GradientDescent(learning_rate=lr, callback=l1_callback, max_iter=1000)
        gd.fit(f=l1, X=None, y=None)

        # l1_weights is a list of the form [array(...), array(...), ...]
        # we want to turn it to np.array of shape (n_iterations, n_features)
        l1_weights = np.array([w for w in l1_weights])

        if gamma == .95:
            contour = plot_descent_path(L1, l1_weights, title=f"L1, eta={eta}, gamma={gamma}")
        if gamma == 1:
            contour.add_trace(go.Scatter(x=l1_weights[:, 0], y=l1_weights[:, 1], mode="markers+lines", marker_color="black", name="gamma = 1"))

        # add the norm of the weights to the figure
        norm_fig.add_trace(go.Scatter(x=np.arange(len(l1_weights)), y=np.linalg.norm(l1_weights, axis=1),
                                        name=f"l1 gamma={gamma}"))

        print(f"min L1, eta={eta}, gamma={gamma}:")
        print(np.min(l1_values))

    norm_fig.update_xaxes(title="Iteration").update_yaxes(title="Norm of Weights").update_layout(title=f"Norm of Weights exponential GD with eta={eta}")
    norm_fig.show()
    contour.show()


def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()

    # convert dfs to numpy arrays
    X_train = X_train.to_numpy(dtype=np.float64)
    y_train = y_train.to_numpy(dtype=np.float64)
    X_test = X_test.to_numpy(dtype=np.float64)
    y_test = y_test.to_numpy(dtype=np.float64)

    # subtract mean from each feature
    # X_test = X_test - X_train.mean(axis=0)
    # X_train = X_train - X_train.mean(axis=0)

    # Plotting convergence rate of logistic regression over SA heart disease data
    logistic_regression = LogisticRegression(solver=GradientDescent(max_iter=20000, learning_rate=FixedLR(1e-4)))
    logistic_regression.fit(X_train, y_train)

    # plot ROC curve
    fpr, tpr, thresholds = roc_curve(y_train, logistic_regression.predict_proba(X_train))

    # get best threshold
    best_threshold = thresholds[np.argmax(tpr - fpr)]
    print(f"Best threshold: {best_threshold}")

    print("Test misclassification error:")
    print(misclassification_error(y_test, logistic_regression.predict(X_test)))

    fig = go.Figure(
        data=[go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="black", dash='dash'),
                         name="Random Class Assignment"),
              go.Scatter(x=fpr, y=tpr, mode='markers+lines', text=thresholds, name="", showlegend=False,
                         marker_size=5,
                         hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")],
        layout=go.Layout(title=rf"$\text{{ROC Curve Of Fitted Model - AUC}}={auc(fpr, tpr):.6f}$",
                         xaxis=dict(title=r"$\text{False Positive Rate (FPR)}$"),
                         yaxis=dict(title=r"$\text{True Positive Rate (TPR)}$")))

    fig.show()


    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter
    lambdas = np.array([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1])
    n_evaluations = len(lambdas)

    l1_train_scores = np.zeros(n_evaluations)
    l1_validation_scores = np.zeros(n_evaluations)

    l2_train_scores = np.zeros(n_evaluations)
    l2_validation_scores = np.zeros(n_evaluations)


    for i, l in enumerate(lambdas):
        l1_logistic = LogisticRegression(solver=GradientDescent(max_iter=20000, learning_rate=FixedLR(1e-4)), lam=l, penalty="l1")
        l1_train_score, l1_validation_score = cross_validate(l1_logistic, X_train, y_train, misclassification_error)
        l1_train_scores[i] = l1_train_score
        l1_validation_scores[i] = l1_validation_score

        l2_logistic = LogisticRegression(solver=GradientDescent(max_iter=20000, learning_rate=FixedLR(1e-4)), lam=l, penalty="l2")
        l2_train_score, l2_validation_score = cross_validate(l2_logistic, X_train, y_train, misclassification_error)
        l2_train_scores[i] = l2_train_score
        l2_validation_scores[i] = l2_validation_score


    best_l1 = LogisticRegression(solver=GradientDescent(max_iter=20000, learning_rate=FixedLR(1e-4)), lam=lambdas[np.argmin(l1_validation_scores)], penalty="l1")
    best_l2 = LogisticRegression(solver=GradientDescent(max_iter=20000, learning_rate=FixedLR(1e-4)), lam=lambdas[np.argmin(l2_validation_scores)], penalty="l2")

    best_l1.fit(X_train, y_train)
    best_l2.fit(X_train, y_train)

    # print misclassification error for best model
    print("best l1 lambda:", lambdas[np.argmin(l1_validation_scores)])
    print(f"Best l1 model misclassification: {best_l1.loss(X_test, y_test)}")
    print("best l2 lambda:", lambdas[np.argmin(l2_validation_scores)])
    print(f"Best l2 model misclassification: {best_l2.loss(X_test, y_test)}")


if __name__ == '__main__':
    np.random.seed(0)
    # compare_fixed_learning_rates()
    # compare_exponential_decay_rates()
    fit_logistic_regression()
