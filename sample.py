import numpy as np

from activations import Linear, Sigmoid
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
loss_function = MSE()
optimizer = Adam(learning_rate)

input_features = X.shape[-1]
output_features = Y.shape[-1]

network = Network(input_size=input_features)
network.add(Layer(shape=(input_features, 3), activation=Linear()))
network.add(Layer(shape=(3, output_features), activation=Sigmoid()))

network.train(x=X, y=Y,
              epochs=epochs,
              batch_size=batch_size,
              loss_function=loss_function,
              optimizer=optimizer,
              shuffle=True)
output = network.predict(X)

sample_range = np.arange(10)
print(np.round(output[sample_range] * 1000) / 1000)
print(Y[sample_range])
