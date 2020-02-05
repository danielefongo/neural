import numpy as np

from neural.activations import Softmax, Tanh
from neural.layers import Layer, SimpleRNN
from neural.losses import CrossEntropy
from neural.networks import Network
from neural.optimizers import GradientDescent

# Load RANDOM data
X = np.random.random_integers(1, 30, (3000, 2, 2))
Y1 = X[:, -1, 0] > X[:, -1, 1]
Y2 = X[:, -1, 0] <= X[:, -1, 1]
Y = np.stack((Y1, Y2), 1) * 1.0

# Train
epochs = 5
batch_size = 32
learning_rate = 0.001

input_features = X.shape[-1]
output_features = Y.shape[-1]

network = Network()
network.add(SimpleRNN(100, 2, Tanh()))
network.add(Layer(Y.shape[-1], Softmax()))
network.train(X, Y, batch_size, epochs, CrossEntropy(), GradientDescent(learning_rate), shuffle=False)

config, variables = network.export()

new_network = Network.use(config, variables)

new_network.train(X, Y, batch_size, epochs, CrossEntropy(), GradientDescent(learning_rate), shuffle=False)
print(new_network.evaluate(X)[:3])
print(Y[:3])

