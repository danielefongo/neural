import numpy as np

from activations import Linear, Sigmoid
from layers import Layer
from losses import MSE, CrossEntropy
from networks import Network
from optimizers import Adam

# Random data
X1 = np.random.random(100)
X2 = np.random.random(100)
Y1 = (X1 > X2) * 1.0
Y2 = ((X1 - X2) > 0.1) * 1.0

Y = np.column_stack((Y1, Y2))
X = np.column_stack((X1, X2))

# Train
epochs = 100
batch_size = 32
learning_rate = 0.01
optimizer = Adam(learning_rate)

input_features = X.shape[-1]
output_features = Y.shape[-1]

network = Network()
network.add(Layer(Linear, shape=(input_features, 200)))
network.add(Layer(Sigmoid, shape=(200, output_features)))

network.train(X, Y,
              batch_size=batch_size,
              epochs=epochs,
              loss_function=CrossEntropy,
              optimizer=Adam(0.01))

print(network.evaluate(X, Y)[:10])
