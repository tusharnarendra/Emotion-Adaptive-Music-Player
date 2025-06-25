import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score
import joblib 

#Load dataset and organize dependent/independent variables
dataset = pd.read_csv('../VA_scores.csv')
X = np.load('../normalized_features.npy')
y = dataset.iloc[:, -2:].values

#Feature scaling
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

#Split dataset into training and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Fit SVR model
regressor = MultiOutputRegressor(SVR(kernel = 'rbf'))
regressor.fit(X_train, y_train)

#Saving the regression model
joblib.dump(regressor, 'svr_multioutput_model.pkl')
joblib.dump(sc_X, 'X_scaler.pkl')
joblib.dump(sc_y, 'y_scaler.pkl')

#Generate predictions
y_pred = regressor.predict(X_test)
y_pred_orig = sc_y.inverse_transform(y_pred)
y_test_orig = sc_y.inverse_transform(y_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred_orig, y_test_orig), axis=1))


#R^2 and adjusted R^2 calculations to evaluate performance
r2 = r2_score(y_test_orig, y_pred_orig)
print("R²:", r2)
n, k = X.shape 
adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)
print("Adjusted R²:", adjusted_r2)