# BLENDED_LERNING
# Implementation-of-Multiple-Linear-Regression-Model-with-Cross-Validation-for-Predicting-Car-Prices

## AIM:
To write a program to predict the price of cars using a multiple linear regression model and evaluate the model performance using cross-validation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import Libraries & Load Dataset
2. Divide the dataset into training and testing sets.
3. Select a suitable ML model, train it on the training data, and make predictions.
4. Assess model performance using metrics and interpret the results.

## Program:
```
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
data= pd.read_csv('CarPrice_Assignment (1).csv')
data.head()

data = data.drop(['car_ID','CarName'], axis=1)
data = pd.get_dummies(data, drop_first=True)
data.head()

X = data.drop('price', axis=1)
y = data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

print('Name: KRITHIKAA P ')
print('Reg. No: 212225040193')
print("\n== Cross-Validation ==")
cv_scores = cross_val_score(model,X,y,cv=5)
print("Fold R^2 scores:", [f"{score:.4f}" for score in cv_scores])
print(f"Average R^2:{cv_scores.mean():.4f}")

y_pred =model.predict(X_test)
print("\n=== Test Set Performance ===")
print(f"MSE: {mean_squared_error(y_test,y_pred):.2f}")
print(f"R^2: {r2_score(y_test,y_pred):.4f}")
print(f"MAE: {mean_absolute_error(y_test,y_pred):>10.2f}")

plt.figure(figsize=(8,6))
plt.scatter(y_test,y_pred,alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Price")
plt.grid(True)
plt.show()
```

## Output:
<img width="1256" height="327" alt="Screenshot 2026-02-23 093238" src="https://github.com/user-attachments/assets/296d16d6-2e7c-426d-8cfd-2160a8aa50f8" />
<img width="1266" height="263" alt="Screenshot 2026-02-23 093248" src="https://github.com/user-attachments/assets/229a7d28-2b0a-448a-84ec-d9327866f0c2" />
<img width="1156" height="89" alt="Screenshot 2026-02-23 093258" src="https://github.com/user-attachments/assets/4a61fcd5-f7a9-4413-805e-c7b25f98a44c" />
<img width="1084" height="146" alt="Screenshot 2026-02-23 093307" src="https://github.com/user-attachments/assets/ae516b97-9c30-4f97-b5b4-197aead3470f" />
<img width="786" height="116" alt="Screenshot 2026-02-23 093313" src="https://github.com/user-attachments/assets/08ad2524-90f1-42ef-9e38-113de579d4b4" />
<img width="1063" height="683" alt="Screenshot 2026-02-23 093323" src="https://github.com/user-attachments/assets/01c39ed0-e701-4d3a-8c74-5702c5c7934a" />



## Result:
Thus, the program to implement the multiple linear regression model with cross-validation for predicting car prices is written and verified using Python programming.
