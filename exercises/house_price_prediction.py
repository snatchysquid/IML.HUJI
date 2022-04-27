from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"

from plotly.subplots import make_subplots


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    df = pd.read_csv(filename)
    df['date'] = df['date'].apply(lambda d: str(d)[:-7])  # get rid of T000000 at the end
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='coerce')

    # get rid of id which is not needed and drop null values
    df = df.drop(columns=["id"]).dropna()

    # get day, month, year
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year

    # drop date column
    df = df.drop(columns=["date"])

    # one hot encoding for zipcode. using corrwith we see that zipcode is correlated with price so we keep it
    df = pd.get_dummies(df, columns=["zipcode"])

    # remove rows with negative price
    df = df[df.price > 0]

    # remove rows where sqft_living is greater than sqft_lot
    df = df[df.sqft_living <= df.sqft_lot]

    # remove rows where sqft_basement is greater than sqft_living
    df = df[df.sqft_basement <= df.sqft_lot]

    # remove rows where there are more than 3 bedrooms but sqft_living is less than 700
    df = df[~((df.bedrooms > 3) & (df.sqft_living < 700))]

    # remove rows where sqft_lot is above 350K (unreasonable for a house)
    # the largest value is 1.5M which is the size of the whole givat ram
    # campus area from the entrace to the national library
    df = df[df.sqft_lot < 350000]
    df = df[df.sqft_lot15 < 250000]

    # remove sqft_lot15 values that are under the minimal value of sqft_lot
    df = df[df.sqft_lot15 >= df.sqft_lot.min()]

    # remove rows where sqft_basement is above 2500
    df = df[df.sqft_basement < 2500]

    # remove rows where bedroom count is greater than 10
    df = df[df.bedrooms < 10]

    # split into X and y
    X = df.drop(columns=["price"])
    y = df["price"]

    # return df

    return X, y

def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """

    # calculate covariance of each feature with y
    covs = X.apply(lambda column: y.cov(column))
    std_y = y.std()
    std_X = X.std()

    pearson = covs / (std_y * std_X)

    # create scatter plot for each feature
    for col in X.columns:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=X[col], y=y, mode='markers', name='data'))\
            .update_layout(title=rf"$\text{{ col-{col} }},\;\rho={np.round(pearson[col], decimals=3)}$")\
            .update_xaxes(title_text=col).update_yaxes(title_text='price')

        fig.write_image(f"{output_path}/{col}.png")


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    X, y = load_data("../datasets/house_prices.csv")

    # Question 2 - Feature evaluation with respect to response

    feature_evaluation(X, y, output_path="./plots")

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(X, y, train_proportion=.75)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)

    lr = LinearRegression()  # init lr model
    full_train = pd.concat([train_X, train_y], axis=1)  # concat train_X and train_y inorder to be able to sample
    losses = np.zeros((91, 10))  # np array to store losses

    for p in range(10, 101):
        for i in range(10):
            # sample p% of the overall training data
            sample_train = full_train.sample(frac=p/100)

            # split to X and y
            X_sample = sample_train.drop(columns=["price"])
            y_sample = sample_train["price"]

            # fit linear model
            lr.fit(X_sample, y_sample)

            # test model and add to losses list
            test_loss = lr._loss(test_X, test_y)
            losses[p-10, i] = test_loss


    # get mean and std of losses
    losses_mean = np.mean(losses, axis=1)
    losses_std = np.std(losses, axis=1)

    x = list(range(10, 101))
    fig = go.Figure(data=(go.Scatter(x=x, y=losses_mean, mode="markers+lines", name="Mean Prediction",
                                     line=dict(dash="dash"), marker=dict(color="green", opacity=.7)),
                          go.Scatter(x=x, y=losses_mean - 2 * losses_std, fill=None, mode="lines",
                                     line=dict(color="lightgrey"), showlegend=False),
                          go.Scatter(x=x, y=losses_mean + 2 * losses_std, fill='tonexty', mode="lines",
                                     line=dict(color="lightgrey"), showlegend=False),),
                    layout=go.Layout(
                        title="Mean loss over test set as a function of train size (in %)",
                        xaxis=dict(title="Train size (%)"),
                        yaxis=dict(title="Mean loss over test set"),
                    ))

    fig.show()


