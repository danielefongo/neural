import numpy as np

X1 = np.random.random_integers(1, 10, 3000)
X2 = np.random.random_integers(1, 10, 3000)
Y = (X1 > X2) * 1

# Train
learning_rate = 0.001
Y = np.expand_dims(Y, 1)
X = np.column_stack((X1, X2, np.ones(X1.size)))

def weighted_sum(w, x):
    return np.matmul(x, w)


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
        self.weights = np.zeros((input_size, output_size))

    def predict(self, x):
        return self.activation.activate(weighted_sum(self.weights, x))

    def update(self, x, out, next_d_error):
        d_w = next_d_error * self.activation.derivative(out)
        gradient = np.matmul(x.T, d_w) / x.shape[0]
        self.weights -= gradient * learning_rate
        return d_w

loss_function = Loss()
layer1 = Layer(X.shape[1], 5, Linear())
layer2 = Layer(5, Y.shape[1], Sigmoid())

for i in np.arange(1, 5000):
    out1 = layer1.predict(X)
    out2 = layer2.predict(out1)
    loss = loss_function.calculate(out2, Y)
    d_loss = loss_function.derivative(out2, Y) # error on output
    print(loss)
    intermediate_d_error = layer2.update(out1, out2, d_loss)
    layer1.update(X, out2, d_loss)

sample_range = np.arange(10)
print(out2[sample_range])
print(Y[sample_range])
