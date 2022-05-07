from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from math import atan2, pi

pio.templates.default = "simple_white"


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    dataset = np.load(filename)

    X = dataset[:, :2]
    y = dataset[:, 2]

    return X, y


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset 
        X, y = load_dataset("../datasets/" + f)

        # Fit Perceptron and record loss in each fit iteration
        losses = []

        def perceptron_callback(fit: Perceptron, sample_x: np.ndarray, sample_y: int):
            # note X, y come from the for loop and they contain the whole dataset
            # the sample_x, sample_y are unused and not needed
            losses.append(fit._loss(X, y))

        max_iter = 200

        perceptron = Perceptron(include_intercept=True, max_iter=max_iter, callback=perceptron_callback)
        perceptron.fit(X, y)

        # Plot figure
        fig = px.line(x=np.arange(len(losses)), y=losses, title=f"Perceptron algorithm on a {n} dataset, with at most {max_iter} iterations") \
            .update_xaxes(title="Iteration") \
            .update_yaxes(title="Misclassification Error")
        fig.show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix
    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse
    cov: ndarray of shape (2,2)
        Covariance of Gaussian
    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (
        np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset("../datasets/" + f)

        # Fit models and predict over training set
        lda = LDA()
        lda.fit(X, y)

        gnb = GaussianNaiveBayes()
        gnb.fit(X, y)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy

        lda_pred = lda.predict(X)
        lda_acc = accuracy(lda_pred, y)

        gnb_pred = gnb.predict(X)
        gnb_acc = accuracy(gnb_pred, y)

        fig = make_subplots(cols=2, subplot_titles=(f"LDA prediction, with accuracy {lda_acc}", f"GNB prediction, with accuracy {gnb_acc}")).update_layout(title=f"Prediction on {f.split('.')[0]} dataset")

        fig.add_trace(
            go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', marker=dict(color=lda_pred),
                       marker_symbol=(lda_pred == y).astype(int)), row=1, col=1)

        # add ellipse
        for _class in range(len(lda.classes_)):
            fig.add_trace(get_ellipse(lda.mu_[_class], lda.cov_), row=1, col=1)
            # add center point
            fig.add_trace(go.Scatter(x=[lda.mu_[_class][0]], y=[lda.mu_[_class][1]], mode="markers",
                                     marker=dict(symbol="x", size=10, color="black")), row=1, col=1)

        fig.add_trace(
            go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', marker=dict(color=gnb_pred),
                       marker_symbol=(gnb_pred == y).astype(int)), row=1, col=2)

        # edit axis labels
        fig['layout']['xaxis']['title'] = 'Feature 1'
        fig['layout']['xaxis2']['title'] = 'Feature 1'
        fig['layout']['yaxis']['title'] = 'Feature 2'
        fig['layout']['yaxis2']['title'] = 'Feature 2'


        # add ellipse
        for _class in range(len(gnb.classes_)):
            fig.add_trace(get_ellipse(gnb.mu_[_class], np.diag(gnb.vars_[_class])), row=1, col=2)
            # add center point
            fig.add_trace(go.Scatter(x=[gnb.mu_[_class][0]], y=[gnb.mu_[_class][1]], mode="markers",
                                     marker=dict(symbol="x", size=10, color="black")), row=1, col=2)

        fig.update_layout(showlegend=False)
        fig.show()


def quiz():
    S = np.array([[0, 0], [1, 0], [2, 1], [3, 1], [4, 1], [5, 1], [6, 2], [7, 2]])
    X = S[:, 0]
    y = S[:, 1]

    gnb = GaussianNaiveBayes()
    gnb.fit(X, y)
    print(np.round(gnb.pi_[0], 2))
    print(np.round(gnb.mu_[1][0], 2))

    S = np.array([[1, 1, 0], [1, 2, 0], [2, 3, 1], [2, 4, 1], [3, 3, 1], [3, 4, 1]])
    X = S[:, :-1]
    y = S[:, -1]

    gnb = GaussianNaiveBayes()
    gnb.fit(X, y)
    print(np.round(gnb.vars_[0, 1], 2))
    print(np.round(gnb.vars_[1, 1], 2))

    S = np.array([(0, 0), (1, 0), (2, 1), (3, 1), (4, 1), (5, 1), (6, 2), (7, 2)])
    print((6+7) / 2)
    print(52)
    print(True)


if __name__ == '__main__':
    np.random.seed(0)
    # run_perceptron()
    # compare_gaussian_classifiers()
    quiz()
