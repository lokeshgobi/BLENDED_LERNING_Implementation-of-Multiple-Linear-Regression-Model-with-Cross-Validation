# BLENDED_LERNING
# Implementation-of-Multiple-Linear-Regression-Model-with-Cross-Validation-for-Predicting-Car-Prices

## AIM:
To write a program to predict the price of cars using a multiple linear regression model and evaluate the model performance using cross-validation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import libraries, load the car price dataset, clean data, and convert categorical values using encoding.
2.Separate features (X) and target (price), then split into training and testing sets.
3.Train the Linear Regression model and evaluate performance using cross-validation.
4.Predict on test data, compute MSE/MAE/R², and plot actual vs predicted prices. 
```
## Program:
```
/*
Program to implement the multiple linear regression model for predicting car prices with cross-validation.
Developed by: LOKESHWARAN.G
RegisterNumber:  212225040210
*/
```
```
import pandas as pd

from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split, cross_val_score 
from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score 
import matplotlib.pyplot as plt

#1. Load and prepare data

#url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML240EN-SkillsNetwork/1

data = pd.read_csv('CarPrice_Assignment (1).csv')
# Simple preprocessing
data = data.drop(['car_ID', 'CarName'], axis=1)
data = pd.get_dummies(data, drop_first=True)
# 2. split data
X = data.drop('price',axis=1)
y = data['price']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
# 3. Create and train model
model = LinearRegression()
model.fit(X_train, y_train)
cv_scores = cross_val_score(model, X, y, cv=5)
# 4. Evalute with cross
print('Name: LOKESHWARAN.G ')
print('reg. No:212225040210 ')
print("\n=== Cross-validation ===")
cv_scores = cross_val_score(model, X, y, cv=5)
print("Fold R² scores:",[f"{score:.4f}" for  score in cv_scores])
print(f"Average R²: {cv_scores.mean():.4f}")
# 5. Test set evaluation
y_pred = model.predict(X_test)
print("\n=== Test set Performance ===")
print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"R²: {r2_score(y_test, y_pred):.4f}")
# 6. Visulization
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices")
plt.grid(True)
plt.show()
```
## Output:

<img width="1061" height="690" alt="image" src="https://github.com/user-attachments/assets/a31ab5ca-7332-407b-a1f9-5bdc286f5ea3" />


## Result:
Thus, the program to implement the multiple linear regression model with cross-validation for predicting car prices is written and verified using Python programming.
