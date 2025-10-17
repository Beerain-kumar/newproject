import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

Create a Simple Dataset

data = {

   'Exam 1 Score': [34, 45, 60, 80, 50, 85, 62, 70, 48, 90],

   'Exam 2 Score': [78, 56, 85, 90, 43, 95, 60, 65, 40, 98],

   'Admitted':     [0, 0, 1, 1, 0, 1, 1, 1, 0, 1]

}

df = pd.DataFrame(data)

print(df)

Prepare Data

X = df[['Exam 1 Score', 'Exam 2 Score']].values

y = df['Admitted'].values

Define Sigmoid Function

def sigmoid(z):

   return 1 / (1 + np.exp(-z))

Initialize Parameters

m, n = X.shape

weights = np.zeros(n)

bias = 0

learning_rate = 0.0005

iterations = 1000

Train Using Gradient Descent

for i in range(iterations):

   linear_model = np.dot(X, weights) + bias

   y_pred = sigmoid(linear_model)

   

   # Cost (log-loss)

   cost = (-1/m) * np.sum(y*np.log(y_pred + 1e-9) + (1-y)*np.log(1 - y_pred + 1e-9))

   

   # Gradients

   dw = (1/m) * np.dot(X.T, (y_pred - y))

   db = (1/m) * np.sum(y_pred - y)

   

   # Update

   weights -= learning_rate * dw

   bias -= learning_rate * db

   

   # Show progress

   if i % 200 == 0:

       print(f"Iteration {i:4d} | Cost: {cost:.4f}")

Prediction Function

def predict(X, weights, bias):

   linear_model = np.dot(X, weights) + bias

   y_pred = sigmoid(linear_model)

   return [1 if i >= 0.5 else 0 for i in y_pred]

 

Evaluate Model

 

y_pred = predict(X, weights, bias)

accuracy = np.mean(y_pred == y) * 100

print("Predictions:", y_pred)

print(f"Accuracy: {accuracy:.2f}%")

 

Visualize Decision Boundary (optional)
plt.figure(figsize=(7,5))

plt.scatter(X[:,0], X[:,1], c=y, cmap='bwr', edgecolors='k')

 

x_values = np.array([min(X[:,0]), max(X[:,0])])

y_values = -(bias + weights[0]*x_values) / weights[1]

 

plt.plot(x_values, y_values, color='green', label='Decision Boundary')

plt.xlabel('Exam 1 Score')

plt.ylabel('Exam 2 Score')

plt.legend()

plt.title('Logistic Regression Decision Boundary')

plt.show()