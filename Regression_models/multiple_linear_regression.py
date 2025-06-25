import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor


dataset = pd.read_csv('../VA_scores.csv')
X = np.load('../normalized_features.npy')
y = dataset.iloc[:, -2:].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

regressor = MultiOutputRegressor(LinearRegression())
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred, y_test), axis=1))
