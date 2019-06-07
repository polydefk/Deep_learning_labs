import numpy as np
import Utils
from matplotlib import pyplot as plt
import time

np.random.seed(123)


class BaseLayer:
    def __init__(self, input_size, output_size, name):
        self.input_size = input_size
        self.output_size = output_size
        self.name = name

    def forward_pass(self, input):
        pass

    def backward_pass(self, grad):
        pass

    def cost(self):
        return 0


class SoftMax(BaseLayer):
    def __init__(self, input_size=0, name='Softmax'):
        super().__init__(input_size, input_size, name)

    def forward_pass(self, input):
        self.X = input
        exp = np.exp(input.T)
        sum = np.sum(exp, axis=0)

        p = (exp / sum).T

        self.p = p

        return p

    def backward_pass(self, labels):
        N = labels.shape[0]
        g = np.empty_like(labels)
        for i in range(N):
            p = self.p[i, :]
            y = labels[i, :]

            g[i, :] = p - y

        return g

    def cost(self, labels, p=None):

        if p is None:
            p = self.p

        l_cross = np.multiply(labels.T, p.T).sum(axis=0)

        J = - np.mean(np.log(l_cross)).sum()

        return J


class ReLU(BaseLayer):
    def __init__(self, input_size=0, name='ReLU'):
        super().__init__(input_size, input_size, name)

    def forward_pass(self, input):
        self.X = input

        h = np.maximum(0, self.X)

        return h

    def backward_pass(self, grad):
        return np.multiply(grad, self.X > 0)

    def cost(self):
        return 0


class Dense(BaseLayer):
    def __init__(self, input_size, output_size, l2_regul, std, name='Dense'):
        super().__init__(input_size, output_size, name)
        self.l2 = l2_regul
        self.std = std

        self.W = np.random.normal(loc=0.0, scale=std, size=(self.output_size, self.input_size))
        self.b = np.zeros((self.output_size, 1))

        self.grad_w = np.zeros_like(self.W, dtype=float)
        self.grad_b = np.zeros_like(self.b, dtype=float)

    def forward_pass(self, X):
        self.X = X

        fwd = (np.dot(self.W, self.X.T) + self.b).T

        return fwd

    def backward_pass(self, grad):
        N = self.X.shape[0]
        grad_w = np.zeros_like(self.W)
        grad_b = np.zeros_like(self.b)

        for i in range(N):
            x = self.X[i, :]
            g = grad[i, :]

            grad_w += np.outer(g, x)
            grad_b += np.reshape(g, grad_b.shape)

        self.grad_w = (grad_w / N) + 2 * np.multiply(self.l2, self.W)
        self.grad_b /= N

        return np.dot(grad, self.W)

    def cost(self):
        powered_weights = np.power(self.W, 2).sum()
        J = np.multiply(self.l2, powered_weights)

        return J


class BnWithScaleShift(BaseLayer):

    def __init__(self, input_size=0, name='BatchNormalization'):
        # WITH SCALE AND SHIFT
        super().__init__(input_size, input_size, name)
        self.epsilon = 1e-16
        self.gamma = np.ones((1, input_size))
        self.beta = np.zeros((1, input_size))

    def forward_pass(self, input):
        self.X = input

        # Batch Norm
        self.bn_fwd = self._normalize()
        # Apply Shift and Scale
        fwd = self.bn_fwd * self.gamma + self.beta
        return fwd

    def backward_pass(self, grad):
        grad = self._bn_scale_shift(grad)
        grad = self._bn_backprop(grad)
        return grad

    def cost(self):
        return 0

    def _normalize(self):
        input = self.X
        N = input.shape[0]

        mu = np.mean(input, axis=0).reshape(1, -1)
        var = np.var(input, axis=0)
        var *= (N - 1) / N

        self.mu, self.var = mu, var

        normalized = np.dot(input - mu, (np.linalg.inv(np.sqrt(np.diag(var + self.epsilon)))))

        return normalized

    def _bn_scale_shift(self, grad):
        N = self.X.shape[0]

        self.grad_gamma = np.dot(np.ones((1, N)), np.multiply(grad, self.bn_fwd)) / N
        self.grad_beta = np.dot(np.ones((1, N)), grad) / N

        grad = np.multiply(np.dot(np.ones((1, N)).T, self.gamma), grad)

        return grad

    def _bn_backprop(self, grad):
        N = self.X.shape[0]

        s1 = np.power((self.var + self.epsilon), -0.5).reshape(1, -1)
        s2 = np.power((self.var + self.epsilon), -1.5).reshape(1, -1)

        grad_1 = np.multiply(np.dot(np.ones((N, 1)), s1), grad)

        grad_2 = np.multiply(np.dot(np.ones((N, 1)), s2), grad)

        D = self.X - np.dot(np.ones((N, 1)), self.mu)
        c = np.dot(np.ones((1, N)), np.multiply(grad_2, D))

        grad = grad_1 - np.dot(np.ones((1, N)), grad_1) / N - np.multiply(np.dot(np.ones((N, 1)), c), D) / N

        return grad


class Classifier(object):
    def __init__(self):
        self.layers = []

    def add_layer(self, layer: BaseLayer):
        self.layers.append(layer)

    def update_weights(self, eta):
        for layer in self.layers:
            if layer.name is 'Dense':
                layer.W -= eta * layer.grad_w
                layer.b -= eta * layer.grad_b

            if layer.name is 'BatchNormalization':
                layer.gamma -= eta * layer.grad_gamma
                layer.beta -= eta * layer.grad_beta

    def forward_pass(self, input):
        output = input
        for layer in self.layers:
            output = layer.forward_pass(output)

        return output

    def backward_pass(self, labels):

        grad = []
        for layer in self.layers[::-1]:
            if layer.name is 'Softmax':
                grad = layer.backward_pass(labels)
            else:
                grad = layer.backward_pass(grad)

    def cost(self, labels, data=None):
        cost = 0
        fwd = None

        if data is not None:
            fwd = self.forward_pass(data)

        for layer in self.layers:
            if layer.name is 'Softmax':
                cost += layer.cost(labels, fwd)
            else:
                cost += layer.cost()

        return cost

    def train_vanila(self, train_data, train_labels, val_data, val_labels, batch_size, n_epochs, eta):
        N = train_data.shape[0]
        n_batch_loops = int(N / batch_size)
        indexes = np.arange(N)
        for i in range(n_epochs):

            for j in range(1, n_batch_loops):
                j_start = (j - 1) * batch_size
                j_end = j * batch_size
                batch_indexes = indexes[j_start: j_end]

                x_batch = train_data[batch_indexes]
                y_batch = train_labels[batch_indexes]

                self.forward_pass(x_batch)
                self.backward_pass(y_batch)
                self.update_weights(eta)

            val_loss = self.cost(val_labels, val_data)
            train_loss = self.cost(train_labels, train_data)
            val_acc = self.predict(val_data, val_labels)
            train_acc = self.predict(train_data, train_labels)

            self.val_loss.append(val_loss)
            self.train_loss.append(train_loss)
            self.val_acc.append(val_acc)
            self.train_acc.append(train_acc)

            print("Epoch={0}, Train_Loss={1}, Val_loss={2}, Train_acc={3}, Val_acc={4}".
                  format(i + 1, round(train_loss, 4), round(val_loss, 4), round(train_acc, 4), round(val_acc, 4)))

    def train_cyclical(self, train_data, train_labels, val_data, val_labels, batch_size, cyclical_values):

        N = train_data.shape[0]
        n_batch_loops = int(N / batch_size)
        indexes = np.arange(N)

        number_of_cycles = cyclical_values['noc']
        eta_min = cyclical_values['eta_min']
        eta_max = cyclical_values['eta_max']
        k = cyclical_values['k']

        l = 0
        time_step = 0
        i = 0
        eta = 0

        ns = k * n_batch_loops
        eta_diff = eta_max - eta_min
        etas = []

        while True:

            for j in range(1, n_batch_loops):

                j_start = (j - 1) * batch_size
                j_end = j * batch_size
                batch_indexes = indexes[j_start: j_end]

                if (2 * l * ns) <= time_step and time_step <= ((2 * l + 1) * ns):

                    eta = eta_min + ((time_step - 2 * l * ns) * eta_diff) / ns

                elif ((2 * l + 1) * ns) <= time_step and time_step <= (2 * (l + 1)) * ns:

                    eta = eta_max - ((time_step - (2 * l + 1) * ns) * eta_diff) / ns
                else:
                    l += 1
                    if l == number_of_cycles:
                        break

                etas.append(eta)
                x_batch = train_data[batch_indexes]
                y_batch = train_labels[batch_indexes]

                self.forward_pass(x_batch)
                self.backward_pass(y_batch)
                self.update_weights(eta)
                time_step += 1

            np.random.shuffle(indexes)
            val_loss = self.cost(val_labels, val_data)
            train_loss = self.cost(train_labels, train_data)
            val_acc = self.predict(val_data, val_labels)
            train_acc = self.predict(train_data, train_labels)

            self.val_loss.append(val_loss)
            self.train_loss.append(train_loss)
            self.val_acc.append(val_acc)
            self.train_acc.append(train_acc)

            i += 1
            print("Train_Loss={0}, Val_loss={1}, Train_acc={2}, Val_acc={3}".
                  format(round(train_loss, 4), round(val_loss, 4), round(train_acc, 4), round(val_acc, 4)))

            if l == number_of_cycles:
                break

        return etas

    def fit(self, train_data, train_labels, val_data, val_labels,
            cyclical_values, n_epochs=100, batch_size=100, eta=0.001):

        self.val_loss = []
        self.val_acc = []
        self.train_loss = []
        self.train_acc = []

        print("            Fit Started!!")
        if cyclical_values:
            print("-----------------------------------------")
            print("        Train Cyclical Version")
            print("-----------------------------------------")
            self.train_cyclical(train_data, train_labels, val_data, val_labels, batch_size, cyclical_values)

        else:
            print("-----------------------------------------")
            print("         Train Vanila Version")
            print("-----------------------------------------")
            self.train_vanila(train_data, train_labels, val_data, val_labels, batch_size, n_epochs, eta)

        return np.array(self.train_loss), np.array(self.val_loss), np.array(self.train_acc), np.array(
            self.val_acc)

    def predict(self, input, labels):
        fwd = self.forward_pass(input)
        pred = np.argmax(fwd.T, axis=0)

        true_indices = np.where(labels == 1)[1]
        correct_pred = np.where(true_indices == pred)[0]

        accuracy = correct_pred.shape[0] * 100 / pred.shape[0]
        round_accuracy = round(accuracy, 3)

        return round_accuracy


if __name__ == "__main__":
    np.seterr(all='raise')
    #
    X_train, y_train, X_val, y_val, X_test, y_test = Utils.load_cifar10(whole_dataset=True)
    X_train, mu, sigma = Utils.normalize_data(X_train)
    X_val, *_ = Utils.normalize_data(X_val, mu, sigma)
    X_test, *_ = Utils.normalize_data(X_test, mu, sigma)

    N = X_train.shape[0]
    d = X_train.shape[1]

    train_examples = N
    dim_size = d

    X_train = X_train[:train_examples, :dim_size]
    y_train = y_train[:train_examples, :dim_size]
    X_val = X_val[:train_examples, :dim_size]
    y_val = y_val[:train_examples, :dim_size]

    noc = 2
    l_min = -5
    l_max = -1
    cyclical_values = {'eta_min': 1e-5, 'eta_max': 1e-1, 'noc': noc, 'k': 5}

    param = {}
    l = 0.005

    start_time = time.time()

    dense1 = Dense(input_size=dim_size, output_size=50, l2_regul=l, std=1 / np.sqrt(dim_size))
    batch_norm1 = BnWithScaleShift(50)
    dense2 = Dense(input_size=50, output_size=50, l2_regul=l, std=1 / np.sqrt(50))
    batch_norm2 = BnWithScaleShift(50)
    dense3 = Dense(input_size=50, output_size=10, l2_regul=l, std=1 / np.sqrt(50))

    model = Classifier()

    model.add_layer(dense1)
    model.add_layer(batch_norm1)
    model.add_layer(ReLU())

    model.add_layer(dense2)
    model.add_layer(batch_norm2)
    model.add_layer(ReLU())

    model.add_layer(dense3)
    model.add_layer(SoftMax())

    train_loss, val_loss, train_acc, val_acc = model.fit(X_train, y_train,
                                                         X_val, y_val,
                                                         cyclical_values=cyclical_values,
                                                         batch_size=100)

    accuracy = model.predict(X_test, y_test)
    print("test accuracy", accuracy)
    end_time = time.time() - start_time

    # param['accuracy on test'] = accuracy
    param['accuracy on validation'] = np.amax(val_acc)
    print("validation maxed accuracy", np.amax(val_acc))

    param['number of cycles'] = noc
    param['time'] = round(end_time, 2)

    # Utils.write_to_csv(param)

    plt.figure()
    plt.title('Cost for lambda: {} cycles: {}'.format(l, noc))
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.legend(['train_loss,val_loss'])

    plt.figure()
    plt.title('Accuracy for lambda: {} cycles: {}'.format(l, noc))
    plt.plot(train_acc)
    plt.plot(val_acc)
    plt.legend(['train_acc,val_acc'])
    plt.show()
