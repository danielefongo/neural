import numpy as np

from activations import Linear, Sigmoid
from losses import CrossEntropy
from networks import Network

# Random data
X1 = np.random.random(3000)
X2 = np.random.random(3000)
Y = (X1 > X2) * 1.0
Y2 = ((X1 - X2) > 0.1) * 1.0

Y = np.column_stack((Y, Y2))
X = np.column_stack((X1, X2, np.ones(X1.size)))

# Train
iterations = 5000
loss_function = CrossEntropy()
learning_rate = 0.1

network = Network(input_size=X.shape[1], learning_rate=learning_rate)
network.add_layer(3, Linear())
network.add_layer(Y.shape[1], Sigmoid())

network.train(x=X, y=Y, iterations=iterations, loss_function=loss_function)
output = network.predict(X)

sample_range = np.arange(10)
print(np.round(output[sample_range] * 1000) / 1000)
print(Y[sample_range])
