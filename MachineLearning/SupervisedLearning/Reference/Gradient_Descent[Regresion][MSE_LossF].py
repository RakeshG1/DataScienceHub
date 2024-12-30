# Import Libraries
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# Sample Data : Uniformly Distributed Random Samples
np.random.seed(101)
X = np.random.uniform(0, 100, 2).astype(int)
y = 3 + (2 * X) 
print("x --> ",X)
print("y --> ",y)

# Visualise Data
# fig = plt.figure(figsize=(10,5))
# sns.scatterplot(x=x, y=y)
# plt.title("Linear Sample Data", fontsize=15, color="black")
# plt.xlabel("x", fontsize=12, color="black")
# plt.ylabel("y", fontsize=12, color="black")
# plt.show()

# Initialise Parameters
w = np.random.randn(1)[0].astype(float).round(2)
b = np.random.randn(1)[0].astype(float).round(2)
print(f"w: {w}, b: {b}")
lr_rate = 0.01 # Learning rate
n_iterations = 100
n_of_samples = len(X)

# Gradient Descent
for iteration in range(n_iterations):
    y_pred = w * X + b
    error = y_pred - y
    print("y --> ", y)
    print("y_pred --> ", y_pred)
    print("error --> ", error)
    gradient_w = (1/n_of_samples) * np.dot(X.T, error)
    print("gradient_w --> ", gradient_w)
    # x -->  [51 57]
    # y -->  [105 117]
    # w: 0.91, b: 0.5
    # y_pred -->  [46.91 52.37]
    # error -->  [-58.09 -64.63]

    # gradient_w -->  -3323.2499999999995
    sys.exit("Testing")
