from metrics.loss_functions import mean_square_error
import numpy as np

y = np.array([1, 2])
y_pred = np.array([8, 2])

print(mean_square_error(y, y_pred))
