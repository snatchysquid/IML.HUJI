import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=['Date']).dropna()

    # add day of the year
    df['DayOfYear'] = df['Date'].dt.dayofyear

    # get rid of rows where temp is under -15
    df = df[df['Temp'] > -15]

    return df


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset

    df = load_data("../datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    israel_df = df[df['Country'] == 'Israel'].copy()

    # create scatter plot where x=dayOfYear, y=temp and the points are colored according to the year
    israel_df['strYear'] = israel_df["Year"].astype(str) #convert to string to make colorscale discrete
    fig = px.scatter(israel_df, x="DayOfYear", y="Temp", color="strYear") \
        .update_layout(title=rf"temperature as a function of the day of the year (with points colored according to the year)") \
        .update_xaxes(title_text="dayOfYear").update_yaxes(title_text='Temp')
    fig.show()

    # groupby month and get std of temp
    israel_df = israel_df[['Temp', 'Month']]
    month_std = israel_df.groupby('Month').agg('std')
    month_std['Month'] = month_std.index

    fig = px.histogram(month_std, x="Month", y='Temp', nbins=12, text_auto=True) \
        .update_xaxes(title_text="Month").update_yaxes(title_text='Temperature STD')\
        .update_layout(title=rf"standard deviation of daily temperature for different months")
    fig.show()


    # Question 3 - Exploring differences between countries
    grouped = df[['Country', 'Month', 'Temp']].groupby(['Country', 'Month']).agg(['std', 'mean']).reset_index()
    grouped.columns = ['Country', 'Month', 'Temp_std', 'Temp_mean']  # more comfortable names

    fig = px.line(grouped, x="Month", y="Temp_mean", line_group="Country", color='Country', error_y="Temp_std") \
            .update_layout(title=rf"Average montly temperature for different countries (with standard deviation)") \
            .update_xaxes(title_text="Month").update_yaxes(title_text='Mean Temperature')
    fig.show()

    # Question 4 - Fitting model for different values of `k`
    israel_df = df[df['Country'] == 'Israel'][['Temp', 'DayOfYear']].copy()

    # split into X and y
    X = israel_df.drop(columns=["Temp"])
    y = israel_df["Temp"]

    train_X, train_y, test_X, test_y = (data.to_numpy().flatten() for data in split_train_test(X, y, train_proportion=.75))
    losses = np.zeros(10)

    for k in range(1, 11):
        polyfit = PolynomialFitting(k)
        polyfit.fit(train_X, train_y)
        losses[k-1] = polyfit.loss(test_X, test_y)
        print(f"k={k}\tloss={losses[k-1]}")

    # create bar plot of losses
    # fig = go.Figure(data=[go.Bar(x=list(range(1, 11)), y=losses)])
    fig = px.bar(x=list(range(1, 11)), y=losses, text_auto=True)\
            .update_layout(title_text="Losses for different values of k")\
            .update_xaxes(title_text="k").update_yaxes(title_text='Loss')

    fig.show()

    # Question 5 - Evaluating fitted model on different countries
    selected_k = 5
    polyfit = PolynomialFitting(selected_k)
    polyfit.fit(israel_df.drop(columns=["Temp"]).to_numpy().flatten(), israel_df["Temp"].to_numpy().flatten())

    coutry_losses = {country: 0 for country in df[df['Country'] != 'Israel']['Country'].unique()}
    for country in coutry_losses.keys():
        country_df = df[df['Country'] == country][['Temp', 'DayOfYear']].copy()
        # split into X and y
        X = country_df.drop(columns=["Temp"]).to_numpy().flatten()
        y = country_df["Temp"].to_numpy().flatten()
        coutry_losses[country] = polyfit.loss(X, y)

    fig = px.bar(x=list(coutry_losses.keys()), y=coutry_losses.values(), text_auto=True)\
            .update_layout(title_text="Losses for different countries")\
            .update_xaxes(title_text="Country").update_yaxes(title_text='Loss')
    fig.show()







