import numpy as np

from neural.activations import Softmax, Tanh
from neural.layers import Layer
from neural.losses import CrossEntropy
from neural.networks import Network
from neural.optimizers import Adam

# Load RANDOM data
X = np.random.random_integers(1, 30, 3000)
X = np.reshape(X, (-1, 2))
Y1 = X[:, 0] > X[:, 1]
Y2 = X[:, 0] <= X[:, 1]
Y = np.stack((Y1, Y2), 1) * 1.0

# Train
epochs = 20
batch_size = 32
learning_rate = 0.001
optimizer = Adam(learning_rate)

input_features = X.shape[-1]
output_features = Y.shape[-1]

network = Network()
network.add(Layer(100, Tanh()))
network.add(Layer(Y.shape[-1], Softmax()))
config = network.export()

new_network = Network()
new_network.use(config)

new_network.train(X, Y, batch_size, epochs, CrossEntropy(), optimizer, shuffle=False)
print(new_network.evaluate(X)[:3])
print(Y[:3])
