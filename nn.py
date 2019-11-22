import numpy as np

X1 = np.random.random_integers(1, 10, 3000)
X2 = np.random.random_integers(1, 10, 3000)
Y = X1 + X2

# Train
learning_rate = 0.001
Y = np.expand_dims(Y, 1)
X = np.column_stack((X1, X2, np.ones(X1.size)))

def weighted_sum(w, x):
    return np.matmul(x, w)


class Activation():  # Linear
    def activate(self, input):
        return input

    def derivative(self, output):
        return 1


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
        return activation_function.activate(weighted_sum(self.weights, x))

    def update(self, x, out, next_d_error):
        d_w = next_d_error * activation_function.derivative(out)
        gradient = np.matmul(x.T, d_w) / x.shape[0]
        self.weights -= gradient * learning_rate
        return d_w

activation_function = Activation()
loss_function = Loss()
layer = Layer(X.shape[1], Y.shape[1], activation_function)

for i in np.arange(1, 1000):
    out = layer.predict(X)
    loss = loss_function.calculate(out, Y)
    d_loss = loss_function.derivative(out, Y) # error on output
    print(loss)
    intermediate_d_error = layer.update(X, out, d_loss)

sample_range = np.arange(10)
print(out[sample_range])
print(Y[sample_range])
