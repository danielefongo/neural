import numpy as np

def weighted_sum(w, x):
    return np.matmul(x, w.T)


class Activation:
    def activate(self, x):
        raise NotImplementedError("Should have implemented this")

    def derivative(self, predicted):
        raise NotImplementedError("Should have implemented this")


class Linear(Activation):
    def activate(self, x):
        return x

    def derivative(self, predicted):
        return np.expand_dims(np.ones(predicted.shape[0]), 1)


class Sigmoid(Activation):
    def activate(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, predicted):
        return predicted * (1 - predicted)


class Loss:
    def calculate(self, predicted, y):
        raise NotImplementedError("Should have implemented this")

    def derivative(self, predicted, y):
        raise NotImplementedError("Should have implemented this")


class MSE(Loss):
    def calculate(self, predicted, y):
        return np.power(predicted - y, 2).mean()

    def derivative(self, predicted, y):
        return 2 * (predicted - y)


class CrossEntropy(Loss):
    def calculate(self, predicted, y):
        first_term = y * np.log(predicted)
        second_term = (1 - y) * np.log(1 - predicted)
        return -1 * np.average(first_term + second_term)

    def derivative(self, predicted, y):
        return predicted - y


class Layer:
    def __init__(self, input_size, output_size, activation):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.weights = np.random.rand(output_size, input_size)

    def predict(self, x):
        return self.activation.activate(weighted_sum(self.weights, x))

    def update(self, x, predicted, d_loss, learning_rate):
        d_activation = d_loss * self.activation.derivative(predicted)
        previous_d_loss = np.matmul(d_activation, self.weights)

        gradient = np.matmul(x.T, d_activation) / x.shape[0]
        self.weights -= gradient.T * learning_rate

        return previous_d_loss


class Network:
    def __init__(self, input_size, learning_rate):
        self.layers = []
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.outputs = []

    def add_layer(self, units, activation):
        if len(self.layers) == 0:
            input_size = self.input_size
        else:
            input_size = self.layers[len(self.layers) - 1].output_size

        self.layers.append(Layer(input_size, units, activation))
        self.outputs.append([])

    def train(self, x, y, iterations, loss_function):
        for i in range(iterations):
            self.predict(x)

            loss, d_loss = self.calculate_loss(y, loss_function)
            print(loss)

            self.backpropagate(x, d_loss)

    def calculate_loss(self, y, loss_function):
        predicted = self._out(len(self.layers) - 1)
        loss_value = loss_function.calculate(predicted, y)
        d_loss = loss_function.derivative(predicted, y)
        return loss_value, d_loss

    def backpropagate(self, x, d_loss):
        for layer_id in np.arange(len(self.layers) - 1, 0, -1):
            d_loss = self._update(layer_id, self.outputs[layer_id - 1], d_loss)

        self._update(0, x, d_loss)

    def predict(self, x):
        self.outputs[0] = self._predict(x, 0)
        for layer_id in np.arange(1, len(self.layers)):
            self.outputs[layer_id] = self._predict(self._out(layer_id - 1), layer_id)

        return self._out(len(self.layers) - 1)

    def _update(self, layer_id, x, d_loss):
        y = self.outputs[layer_id]
        return self.layers[layer_id].update(x, y, d_loss, self.learning_rate)

    def _out(self, layer_id):
        return self.outputs[layer_id]

    def _predict(self, x, layer_id):
        return self.layers[layer_id].predict(x)


# Random data
X1 = np.random.random(3000)
X2 = np.random.random(3000)
Y = (X1 > X2) * 1.0
Y2 = ((X1 - X2) > 0.1) * 1.0

Y = np.column_stack((Y, Y2))
X = np.column_stack((X1, X2, np.ones(X1.size)))

# Train
iterations = 10000
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
