# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. **Import the necessary libraries and load the dataset.**  
   - Import `numpy`, `pandas`, and `sklearn.preprocessing.StandardScaler`.
   - Read the `50_Startups.csv` dataset using pandas.
   - Display the first few rows of the dataset to understand the data.

2. **Prepare the feature set and target variable.**  
   - Extract the input features `X` (all columns except the last two).
   - Extract the target variable `y` (the last column).
   - Convert features and target to float type.
   - Standardize the features and the target using `StandardScaler`.

3. **Define the Linear Regression function using Gradient Descent.**  
   - Add a bias (intercept) column of ones to the feature matrix `X`.
   - Initialize parameters (`theta`) with zeros.
   - Perform the specified number of iterations:
     - Compute predictions using the current `theta`.
     - Calculate the error between predictions and actual values.
     - Update `theta` using the gradient descent formula.

4. **Train the model using the training data.**  
   - Pass the standardized feature set and target values to the `linear_regression` function.
   - Obtain the optimized parameters (`theta`) after training.

5. **Make predictions for new data and inverse transform the result.**  
   - Standardize the new input data point.
   - Add the bias term and predict the scaled output using the trained model.
   - Inverse transform the predicted value back to the original scale to get the final result.

---

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Abishek Priyan M
RegisterNumber:  212224240004
*/
```
```py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate=0.01,num_iters=1000):
    X=np.c_[np.ones(len(X1)),X1]
    theta=np.zeros(X.shape[1]).reshape(-1,1)
    #perform gradient descent
    for _ in range(num_iters):
        predictions=(X).dot(theta).reshape(-1,1)
        #calulate errors
        errors=(predictions-y).reshape(-1,1)
        #update theta using gradient descent
        theta-=learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
data=pd.read_csv('50_Startups.csv')
print(data.head())

#Assuming the last column is your target value
X=(data.iloc[1:,:-2].values)
print(X)

X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
print(y)

X1_scaled=scaler.fit_transform(X1)
Y1_scaled=scaler.fit_transform(y)
print(X1_scaled)
print(Y1_scaled)

#Learn model parameters
theta=linear_regression(X1_scaled,Y1_scaled)

#predict taget value for new data point
new_data=np.array([165349.2,136897.8,471784.1])
new_scaled=scaler.fit_transform(new_data)
new_scaled=np.dot(np.append(1,new_scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted value: {pre}")
```
## Output:

## Dataset
![image](https://github.com/user-attachments/assets/e4b27c9a-e4b5-4670-97c8-6f7c864af697)

## X value
![image](https://github.com/user-attachments/assets/d0863caa-56a5-41ee-bd23-92306f5e1a9c)

## Y Value
![image](https://github.com/user-attachments/assets/4d9e0090-7260-40d8-a1ab-3df024bdbe64)

## X1_scaled
![image](https://github.com/user-attachments/assets/fd76ee14-dfe9-4f41-b313-593a6d1bdca3)

## Y1_scaled
![image](https://github.com/user-attachments/assets/39e9e15e-d7cb-4cbe-9727-3569691eaafb)

## Predicted Value
![outpu_ml](https://github.com/user-attachments/assets/a9c1b1ea-7ed1-4e55-af8f-ccbd3f587b21)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
