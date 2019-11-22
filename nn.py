import numpy as np


X1 = np.random.random_integers(1, 10, 3000)
X2 = np.random.random_integers(1, 10, 3000)
Y = X1 + X2

#Train
learning_rate = 0.001
Y = np.expand_dims(Y, 1)
X = np.column_stack((X1, X2, np.ones(X1.size)))
W = np.zeros((X.shape[1], Y.shape[1]))


def predict(x, w):
    return np.matmul(x, w)

def calculate_loss(y, out):
    return np.power(out - y, 2).mean()

def calculate_gradient(x, y, out):
    return (2 * np.matmul(x.T, out - y)) / x.shape[0]

for i in np.arange(1, 1000):
    out = predict(X, W)
    loss = calculate_loss(Y, out)
    print(loss)
    gradient = calculate_gradient(X, Y, out)
    W -= gradient * learning_rate

sample_range = np.arange(10)
print(out[sample_range])
print(Y[sample_range])
