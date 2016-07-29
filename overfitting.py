import numpy as np
import statsmodels.formula.api as smf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


# Set seed for reproducible results
np.random.seed(414)

# Gen toy data
X = np.linspace(0, 15, 1000)
y = 3 * np.sin(X) + np.random.normal(1 + X, .2, 1000)

train_X, train_y = X[:700], y[:700]
test_X, test_y = X[700:], y[700:]

train_df = pd.DataFrame({'X': train_X, 'y': train_y})
test_df = pd.DataFrame({'X': test_X, 'y': test_y})

# Polynomial n=1 train
poly_1 = smf.ols(formula='y ~ 1 + X', data=train_df).fit()
# Polynomial n=2 train
poly_2 = smf.ols(formula='y ~ 1 + X + I(X**2)', data=train_df).fit()
# Polynomial n=3 train
poly_3 = smf.ols(formula='y ~ 1 + X + I(X**2) + I(X**3)', data=train_df).fit()
# Polynomial n=4 train
poly_4 = smf.ols(formula='y ~ 1 + X + I(X**2) + I(X**3) + I(X**4)', data=train_df).fit()

# Polynomial Fit n=1
pred1 = poly_1.predict(test_df['X'])[700:]
plt.plot(test_df['X'], test_df['y'], 'o')
plt.plot(test_df['X'], pred1, 'r')
# Polynomial Fit n=2
pred2 = poly_2.predict(test_df['X'])[700:]
plt.plot(test_df['X'], pred2, 'g')
# Polynomial Fit n=3
pred3 = poly_3.predict(test_df['X'])[700:]
plt.plot(test_df['X'], pred3, 'b')
# Polynomial Fit n=4
pred4 = poly_4.predict(test_df['X'])[700:]
plt.plot(test_df['X'], pred4, 'y')

# Mean Squared Error Calculations
mse1 = mean_squared_error(test_df['y'], pred1)
mse2 = mean_squared_error(test_df['y'], pred2)
mse3 = mean_squared_error(test_df['y'], pred3)
mse4 = mean_squared_error(test_df['y'], pred4)

# Round down the MSE values
mse_str1 = "{:.3f}".format(mse1)
mse_str2 = "{:.3f}".format(mse2)
mse_str3 = "{:.3f}".format(mse3)
mse_str4 = "{:.3f}".format(mse4)

# Printed output
print(poly_1.summary())
print(poly_2.summary())
print(poly_3.summary())
print(poly_4.summary())
print("Polynomial Fit n=1 MSE = {}".format(mse1))
print("Polynomial Fit n=2 MSE = {}".format(mse2))
print("Polynomial Fit n=3 MSE = {}".format(mse3))
print("Polynomial Fit n=4 MSE = {}".format(mse4))

# Label and save graph
plt.xlabel("Test X")
plt.ylabel("Test Y")
plt.title("Prediction Analysis")
plt.savefig('./graphs/predicted.png')
