import numpy as np


X1 = np.random.random_integers(1, 10, 3000)
X2 = np.random.random_integers(1, 10, 3000)
Y = X1 + X2

#Train
learning_rate = 0.001
Y = np.expand_dims(Y, 1)
X = np.column_stack((X1, X2, np.ones(X1.size)))
W = np.zeros((X.shape[1], Y.shape[1]))


def weighted_sum(w, x):
    return np.matmul(x, w)

def activate(input):
    return input

def derivative_activate(output):
    return 1

def predict(x, w):
    return activate(weighted_sum(w, x))

def calculate_loss(y, out):
    return np.power(out - y, 2).mean()

def derivative_loss(y, out):
    return (2 * (out - y))

def calculate_gradient(x, y, out):
    d_loss = derivative_loss(y, out)
    d_w = d_loss * derivative_activate(out)
    return np.matmul(x.T, d_w) / x.shape[0]

for i in np.arange(1, 1000):
    out = predict(X, W)
    loss = calculate_loss(Y, out)
    print(loss)
    gradient = calculate_gradient(X, Y, out)
    W -= gradient * learning_rate

sample_range = np.arange(10)
print(out[sample_range])
print(Y[sample_range])
