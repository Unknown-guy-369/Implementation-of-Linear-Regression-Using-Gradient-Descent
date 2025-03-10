# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Import the required library and read the dataframe.

2. Write a function computeCost to generate the cost function.

3. Perform iterations og gradient steps with learning rate.

4. Plot the Cost function using Gradient Descent and generate the required graph.

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
