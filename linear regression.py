import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
x_train = np.array([1.0, 1.7, 2.0, 2.5, 3.0, 3.2, 4.1, 5.2])  # Size in 100 square feet
y_train = np.array([250, 300, 480, 430, 630, 730, 810, 880])  # House price in 1000$


def compute_gradient(x, y, w, b):  # F fucntions to compute gradient descent(finding derivative of w and b)

    # Number of training examples
    m = len(x) # taking the length for the iteration
    dw = 0
    db = 0
    for i in range(m):
        dw += (w * x[i] + b - y[i]) * x[i]  # calculating derivative of w
        db += (w * x[i] + b - y[i])  # calculating deribative of b
    dw = dw / m
    db = db / m
    return dw, db


def gradient_descent(x, y, w, b, alpha, max_iters):

    for i in range(max_iters):
        dw, db = compute_gradient(x, y, w, b)
        b = b - alpha * db
        w = w - alpha * dw
    return w, b


def compute_model_output(x, w, b):  # functions for the line f = wx +b
    m = len(x)
    #take a list for y value
    predicterd_y=[]
    for i in range(m):
        predicterd_y.append ( w * x[i] + b )
    return predicterd_y



finalW, finalB = gradient_descent(x_train, y_train, 0, 0, .001, 100000) # x , y , w , b, alpha , iteration

print(f'w_final = {finalW} and b_final = {finalB}')  # final value of w and b for best fit

predicetedY = compute_model_output(x_train, finalW, finalB)  # f points to display the dataset
plt.scatter(x_train, y_train, marker='*', c='r')  # ploting the data points
# Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')
plt.plot(x_train, predicetedY)  # ploting the line
plt.legend()
plt.show()


