import numpy as np

from activations import Sigmoid
from layers import Layer
from losses import MSE
from networks import Network
from optimizers import Adam

# Random data

X1 = np.random.random(3000)
X2 = np.random.random(3000)
Y1 = (X1 > X2) * 1.0
Y2 = ((X1 - X2) > 0.1) * 1.0

Y = np.column_stack((Y1, Y2))
X = np.column_stack((X1, X2))

# Train
epochs = 50
batch_size = 32
learning_rate = 0.01
optimizer = Adam(learning_rate)

input_features = X.shape[-1]
output_features = Y.shape[-1]

network = Network()
network.add(Layer(2).copy())
network.add(Layer(output_features, activation=Sigmoid()))
network.train(X, Y, batch_size, epochs, MSE(), optimizer)

print(network.unit.evaluate()[:10])
