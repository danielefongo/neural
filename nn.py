import numpy as np

X1 = np.random.random(3000)
X2 = np.random.random(3000)
Y = (X1 > X2) * 1.0
Y2 = ((X1 - X2) > 0.1) * 1.0

# Train
Y = np.column_stack((Y, Y2))
X = np.column_stack((X1, X2, np.ones(X1.size)))

def weighted_sum(w, x):
    return np.matmul(x, w.T)


class Linear():  # Linear
    def activate(self, input):
        return input

    def derivative(self, output):
        return np.expand_dims(np.ones(output.shape[0]), 1)

class Sigmoid():  # Linear
    def activate(self, input):
        return 1 / (1 + np.exp(-input))

    def derivative(self, output):
        return output * (1 - output)


class Loss():
    def calculate(self, out, y):
        return np.power(out - y, 2).mean()

    def derivative(self, out, y):
        return (2 * (out - y))


class Layer():
    def __init__(self, input_size, output_size, activation):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.weights = np.random.rand(output_size, input_size)

    def predict(self, x):
        return self.activation.activate(weighted_sum(self.weights, x))

    def update(self, x, out, next_loss_d, learning_rate):
        d_w = next_loss_d * self.activation.derivative(out)
        previous_loss_d = np.matmul(d_w, self.weights)

        gradient = np.matmul(x.T, d_w) / x.shape[0]
        self.weights -= gradient.T * learning_rate
        return previous_loss_d

class Network():
    def __init__(self, input_size, learning_rate):
        self.layers = []
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.outputs = []

    def addLayer(self, units, activation):
        if len(self.layers) == 0:
            input_size = self.input_size
        else:
            input_size = self.layers[len(self.layers) - 1].output_size

        print(input_size)
        self.layers.append(Layer(input_size, units, activation))
        self.outputs.append([])

    def train(self, iterations, X, Y, loss):
        for i in range(iterations):
            self.predict(X)

            lastLayer = self.outputs[len(self.layers) - 1]
            loss_value = loss.calculate(lastLayer, Y)
            d_loss = loss.derivative(lastLayer, Y)
            print(loss_value)

            self.backpropagate(X, d_loss)

    def backpropagate(self, X, d_loss):
        for layer_id in np.arange(len(self.layers) - 1, 0, -1):
            d_loss = self._update(layer_id, self.outputs[layer_id - 1], d_loss)

        self._update(0, X, d_loss)

    def predict(self, X):
        self.outputs[0] = self._predict(X, 0)
        for layer_id in np.arange(1, len(self.layers)):
            self.outputs[layer_id] = self._predict(self._out(layer_id - 1), layer_id)

        return self._out(len(self.layers) - 1)

    def _update(self, layer_id, X, d_loss):
        Y = self.outputs[layer_id]
        return self.layers[layer_id].update(X, Y, d_loss, self.learning_rate)

    def _out(self, layer_id):
        return self.outputs[layer_id]

    def _predict(self, X, layer_id):
        return self.layers[layer_id].predict(X)

iterations = 10000
loss = Loss()
learning_rate = 0.1

network = Network(input_size=X.shape[1], learning_rate=learning_rate)
network.addLayer(3, Linear())
network.addLayer(Y.shape[1], Sigmoid())

network.train(iterations=iterations, X=X, Y=Y, loss=loss)
out = network.predict(X)

sample_range = np.arange(10)
print(np.round(out[sample_range] * 1000) / 1000)
print(Y[sample_range])
