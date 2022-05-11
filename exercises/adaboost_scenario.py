import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    adaboost = AdaBoost(DecisionStump, n_learners)
    adaboost.fit(train_X, train_y)

    losses = np.empty((n_learners, 2))
    for i in range(n_learners):
        losses[i, 0] = adaboost.partial_loss(train_X, train_y, i+1)
        losses[i, 1] = adaboost.partial_loss(test_X, test_y, i+1)

    fig = go.Figure().update_layout(
        title="AdaBoost Train- and Test Errors"
    )
    fig.add_scatter(x=np.arange(1, n_learners+1), y=losses[:, 0], name='Train error')
    fig.add_scatter(x=np.arange(1, n_learners+1), y=losses[:, 1], name='Test error')
    fig.update_layout(xaxis_title='# of learners', yaxis_title='Error')

    fig.show()


    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])

    # combine train and test data
    X = np.r_[train_X, test_X]
    symbols = np.array(["circle", "x"])

    fig = make_subplots(rows=2, cols=2, subplot_titles=[rf"$\textbf{{{m}}}$" for m in T],
                        horizontal_spacing=0.01, vertical_spacing=.03)
    for i, t in enumerate(T):
        fig.add_traces([decision_surface(lambda X_samples: adaboost.partial_predict(X_samples, t), lims[0], lims[1], showscale=False),
                        go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers", showlegend=True, name=f"Train - {t}",
                                   marker=dict(color=train_y, symbol=symbols[0], colorscale=[custom[0], custom[-1]],
                                               line=dict(color="black", width=1))),
                        go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=True, name=f"Test - {t}",
                                   marker=dict(color=test_y, symbol=symbols[1], colorscale=[custom[0], custom[-1]],
                                               line=dict(color="black", width=1)))
                        ],
                       rows=(i // 2) + 1, cols=(i % 2) + 1)

    fig.update_layout(title=rf"$\textbf{{Decision Boundaries Of The Same Model With Different Ensemble Size}}$",
                      margin=dict(t=100)) \
        .update_xaxes(visible=False).update_yaxes(visible=False)


    fig.show()

    # Question 3: Decision surface of best performing ensemble
    best_t = np.argmin(losses[:, 1]) + 1
    fig = go.Figure()
    fig.add_traces([decision_surface(lambda X_samples: adaboost.partial_predict(X_samples, t), lims[0],
                                     lims[1], showscale=False),
                    go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers", showlegend=True,
                               name=f"Train",
                               marker=dict(color=train_y, symbol=symbols[0],
                                           colorscale=[custom[0], custom[-1]],
                                           line=dict(color="black", width=1))),
                    go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=True,
                               name=f"Test",
                               marker=dict(color=test_y, symbol=symbols[1],
                                           colorscale=[custom[0], custom[-1]],
                                           line=dict(color="black", width=1)))
                    ])

    fig.update_layout(title=rf"$\textbf{{Decision Boundaries Of The Best Ensemble - T = {best_t} }}$",
                      margin=dict(t=100)) \
        .update_xaxes(visible=False).update_yaxes(visible=False)

    fig.show()

    # Question 4: Decision surface with weighted samples
    D = adaboost.D_ / np.max(adaboost.D_)
    D *= 50

    fig = go.Figure()
    fig.add_traces([decision_surface(adaboost.predict, lims[0], lims[1], showscale=False),
                    go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers", showlegend=True,
                               name=f"Train",
                               marker=dict(color=train_y, symbol=symbols[0],
                                           colorscale=[custom[0], custom[-1]],
                                           line=dict(color="black", width=1),
                                           size=D)),
                    ])

    fig.update_layout(title=rf"$\textbf{{Decision Boundaries Of The Best Ensemble - Weighted Samples}}$",
                      margin=dict(t=100)) \
        .update_xaxes(visible=False).update_yaxes(visible=False)

    fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(noise=0)

