from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
import numpy as np
from typing import Tuple
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.express as px

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
    for n, f in [("Linearl                                            y Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset 
        X, y = load_dataset("../datasets/" + f)

        # Fit Perceptron and record loss in each fit iteration
        losses = []

        def perceptron_callback(fit: Perceptron, sample_x: np.ndarray, sample_y: int):
            # note X, y come from the for loop and they contain the whole dataset
            # the sample_x, sample_y are unused and not needed
            losses.append(fit._loss(X, y))

        perceptron = Perceptron(include_intercept=True, max_iter=100, callback=perceptron_callback)
        perceptron.fit(X, y)

        # Plot figure
        fig = px.line(x=np.arange(len(losses)), y=losses, title=f"Perceptron algorithm on a {n} dataset") \
            .update_xaxes(title="Iteration") \
            .update_yaxes(title="Misclassification Error")
        fig.show()


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
        from IMLearn.metrics import accuracy

        lda_pred = lda.predict(X)
        lda_acc = accuracy(lda_pred, y)

        gnb_pred = gnb.predict(X)
        gnb_acc = accuracy(gnb_pred, y)

        fig = make_subplots(rows=2).update_layout(title=f"LDA prediction on a {f.split('.')[0]} dataset")
        fig.add_trace(
            go.Scatter(x=X[:, 0], y=X[:, 1], marker=lda_pred, name=f"LDA prediction, with accuracy {lda_acc}") \
                .update_xaxes(title="Feature 1") \
                .update_yaxes(title="Feature 2"), row=1, col=1)




def test():
    X, y = load_dataset("../datasets/" + "gaussian1.npy")
    likely_X = X[-5:]
    likely_y = y[-5:]
    X = X[:15]
    y = y[:15]

    # LDA SHIT
    print("LDA shit")
    lda = LDA()
    lda.fit(X, y)
    print(lda.predict(X))
    print(lda.likelihood(likely_X))

    # print lda attributes
    print(lda.mu_)
    print(lda.cov_)
    print(lda.pi_)

    # Gaussian Naive Bayes SHIT
    print("GNB shit")
    gnb = GaussianNaiveBayes()
    gnb.fit(X, y)
    print(gnb.predict(X))
    print(gnb.likelihood(likely_X))

    # print gnb attributes
    print(gnb.mu_)
    print(gnb.vars_)
    print(gnb.pi_)


if __name__ == '__main__':
    np.random.seed(0)
    # run_perceptron()
    # compare_gaussian_classifiers()
    test()
