import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

dataset = pd.read_csv('../VA_scores.csv')
X = np.load('../normalized_features.npy')
y = dataset.iloc[:, -2:].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regressor = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=0))
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

np.set_printoptions(precision=2)
print(np.concatenate((y_pred, y_test), axis=1))

#R^2 and adjusted R^2 calculations to evaluate performance
r2 = r2_score(y_test, y_pred)
print("R²:", r2)
n, k = X.shape 
adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)
print("Adjusted R²:", adjusted_r2)