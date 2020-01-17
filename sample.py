import numpy as np

from activations import Linear
from layers import SimpleRNN
from losses import MSE
from networks import Network
from optimizers import Adam

# Random data

X = np.array([[[1], [2], [3]], [[2], [3], [4]]])
Y = X[:, -1] + 1

# Train
epochs = 200
batch_size = 2
learning_rate = 0.01
optimizer = Adam(learning_rate)

input_features = X.shape[-1]
output_features = Y.shape[-1]

network = Network()
network.add(SimpleRNN(1, X.shape[1], Linear()))
network.train(X, Y, batch_size, epochs, MSE(), optimizer, shuffle=False)

print(network.unit.evaluate())
