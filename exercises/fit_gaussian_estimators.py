from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import pandas as pd
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    MU = 10
    VAR = 1

    # Question 1 - Draw samples and print fitted model

    # get random samples
    X = np.random.normal(MU, VAR, size=1000)

    # fit to a gaussian
    uni_est = UnivariateGaussian()
    uni_est.fit(X)

    # print the parameters
    print(uni_est.mu_, uni_est.var_)


    # Question 2 - Empirically showing sample mean is consistent
    expectations = np.zeros(100)
    for ind, i in enumerate(range(10, 1010, 10)):
        uni_est.fit(X[:i])
        expectations[ind] = uni_est.mu_

    expectations -= MU

    # fig = px.scatter(x=list(range(10, 1010, 10)), y=np.abs(expectations),
    #                  title="Empirical sample mean converging to true mean",
    #                  labels={
    #                      "x": "Sample size",
    #                      "y": "sample & true mean abs distance"
    #                  })
    # fig.show()


    # Question 3 - Plotting Empirical PDF of fitted model
    X = np.sort(X)
    uni_est.fit(X)

    fig = px.scatter(x=X, y=uni_est.pdf(X),
                     title="Empirical PDF of fitted model",
                     labels={
                         "x": "sample value",
                         "y": "sample pdf"
                     })
    fig.show()


def test_multivariate_gaussian():
    MU = np.array([0, 0, 4, 0])
    SIGMA = np.array([
        [1, 0.2, 0, 0.5],
        [0.2, 2, 0, 0],
        [0, 0, 1, 0],
        [0.5, 0, 0, 1]
    ])

    # Question 4 - Draw samples and print fitted model
    X = np.random.multivariate_normal(MU, SIGMA, size=1000)

    # fit to a gaussian
    multi_est = MultivariateGaussian()
    multi_est.fit(X)

    # print the parameters
    # print(multi_est.mu_)
    # print(multi_est.cov_)


    # Question 5 - Likelihood evaluation
    f1 = np.linspace(-10, 10, 200)
    org_mu_values = cartesian_product(f1, f1)

    # insert two zeros - one after each element
    mu_values = np.insert(org_mu_values, 1, 0, axis=1)
    mu_values = np.insert(mu_values, 3, 0, axis=1)

    result = np.zeros(mu_values.shape[0])
    # calculate the likelihood for each mu value

    import time  # Q5
    start = time.time()

    for i in range(len(mu_values)):
        result[i] = multi_est.log_likelihood(mu_values[i], SIGMA, X)

    end = time.time()
    print(end - start)   # 2.0545549392700195

    print(org_mu_values[np.argmax(result)])


    # plot heatmap
    fig = px.density_heatmap(x=org_mu_values[:, 0], y=org_mu_values[:, 1], z=result,
                             title="Log likelihood of each mu value",
                             labels={
                                 "x": "mu1",
                                 "y": "mu2",
                                 "z": "log likelihood"
                             })
    fig.show()


    # Question 6 - Maximum likelihood
    raise NotImplementedError()

###########
# UTILITY #
###########


def cartesian_product(vec1, vec2):
    # np.repeat([1, 2, 3], 4) -> [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
    # np.tile([1, 2, 3], 4)   -> [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]
    return np.transpose(np.array([np.repeat(vec1, len(vec2)), np.tile(vec2, len(vec1))]))


if __name__ == '__main__':
    np.random.seed(0)
    # test_univariate_gaussian()
    test_multivariate_gaussian()
