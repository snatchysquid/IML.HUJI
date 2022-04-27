from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

response = lambda x: x ** 4 - 2 * x ** 3 - .5 * x ** 2 + 1

x = np.linspace(-1.2, 2, 30)
y_ = response(x)

polynomial_degree = 8
frames, preds = [], []
for _ in range(10):
    y = y_ + np.random.normal(scale=2, size=len(y_))
    y_hat = make_pipeline(PolynomialFeatures(polynomial_degree), LinearRegression()).fit(x.reshape(-1, 1),
                                                                                         y).predict(
        x.reshape(-1, 1))
    preds.append(y_hat)

    frames.append(go.Frame(
        data=[
            go.Scatter(x=x, y=y_, mode="markers+lines", name="Real Points",
                       marker=dict(color="black", opacity=.7)),
            go.Scatter(x=x, y=y, mode="markers", name="Observed Points",
                       marker=dict(color="red", opacity=.7)),
            go.Scatter(x=x, y=y_hat, mode="markers+lines", name="Predicted Points",
                       marker=dict(color="blue", opacity=.7))],
        layout=go.Layout(
            title_text=rf"$\text{{Polynomial Fitting of Degree {polynomial_degree} - Sample Noise }}\mathcal{{N}}\left(0,2\right)$",
            xaxis={"title": r"$x$"},
            yaxis={"title": r"$y$", "range": [-6, 10]})))

mean_pred, var_pred = np.mean(preds, axis=0), np.var(preds, axis=0)
for i in range(len(frames)):
    frames[i]["data"] = (go.Scatter(x=x, y=mean_pred, mode="markers+lines", name="Mean Prediction",
                                    line=dict(dash="dash"), marker=dict(color="green", opacity=.7)),
                         go.Scatter(x=x, y=mean_pred - 2 * var_pred, fill=None, mode="lines",
                                    line=dict(color="lightgrey"), showlegend=False),
                         go.Scatter(x=x, y=mean_pred + 2 * var_pred, fill='tonexty', mode="lines",
                                    line=dict(color="lightgrey"), showlegend=False),) + frames[i]["data"]

fig = go.Figure(data=frames[0]["data"],
                frames=frames[1:],
                layout=go.Layout(
                    title=frames[0]["layout"]["title"],
                    xaxis=frames[0]["layout"]["xaxis"],
                    yaxis=frames[0]["layout"]["yaxis"],
                    updatemenus=[dict(visible=True,
                                      type="buttons",
                                      buttons=[dict(label="Play",
                                                    method="animate",
                                                    args=[None, dict(frame={"duration": 1000})])])]))

animation_to_gif(fig, f"../poly-deg{polynomial_degree}-diff-samples.gif", 1000)
fig.show()