import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import time
import os
import csv
from tqdm import tqdm
from copy import deepcopy
import pickle

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # Directory of the script
CIFAR_DIR = os.path.join(ROOT_DIR, 'CIFAR10/')  # Directory of the dataset
PLOT_DIR = os.path.join(ROOT_DIR, 'plots/')
np.random.seed(123)


def write_to_csv(dictionary):
    with open('results.csv', 'a', newline='') as csvfile:
        fieldnames = dictionary.keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        # writer.writeheader()
        writer.writerow(dictionary)


def grad_checking_both_layers():
    X_train, y_train, X_val, y_val, X_test, y_test = Utils.load_cifar10(whole_dataset=False)
    batch_size = 20
    dim_size = 500
    X_train, mu, sigma = Utils.normalize_data(X_train)
    X_val, *_ = Utils.normalize_data(X_val, mu, sigma)
    X_test, *_ = Utils.normalize_data(X_test, mu, sigma)

    X_train = X_train[:batch_size, :dim_size]
    y_train = y_train[:batch_size]

    dense1 = Dense(input_size=dim_size, output_size=50, l2_regul=0., std=1 / np.sqrt(dim_size))
    dense2 = Dense(input_size=50, output_size=10, l2_regul=0, std=1 / np.sqrt(50))

    model = Classifier()
    model.add_layer(dense1)
    model.add_layer(ReLU())
    model.add_layer(dense2)
    model.add_layer(SoftMax())

    model.forward_pass(X_train)
    model.backward_pass(y_train)

    weights = [dense1.W, dense2.W]

    grad_w = [dense1.grad_w, dense2.grad_w]

    print("\n\n\n===========================weigths===========================")
    compute_grads_for_matrix(y_train, X_train, weights, model, grad_w)


def load_configurations():
    configurations = []
    configurations.append({'loss_type': 'cross_entropy',
                           'l2': 0., 'epochs': 40, 'batch_size': 100, 'eta': 0.1, 'shuffle': False, 'decay': False})
    configurations.append({'loss_type': 'cross_entropy',
                           'l2': 0., 'epochs': 40, 'batch_size': 100, 'eta': 0.01, 'shuffle': False, 'decay': False})
    configurations.append({'loss_type': 'cross_entropy',
                           'l2': 0.1, 'epochs': 40, 'batch_size': 100, 'eta': 0.01, 'shuffle': False, 'decay': False})
    configurations.append({'loss_type': 'cross_entropy',
                           'l2': 1, 'epochs': 40, 'batch_size': 100, 'eta': 0.01, 'shuffle': False, 'decay': False})
    configurations.append({'loss_type': 'svm',
                           'l2': 0.01, 'epochs': 50, 'batch_size': 100, 'eta': 0.01, 'shuffle': False, 'decay': False})
    configurations.append({'loss_type': 'cross_entropy',
                           'l2': 0.01, 'epochs': 50, 'batch_size': 100, 'eta': 0.01, 'shuffle': False, 'decay': False})
    configurations.append({'loss_type': 'svm',
                           'l2': 1, 'epochs': 50, 'batch_size': 100, 'eta': 0.1, 'shuffle': False, 'decay': False})
    configurations.append({'loss_type': 'cross_entropy',
                           'l2': 1, 'epochs': 50, 'batch_size': 100, 'eta': 0.1, 'shuffle': False, 'decay': False})
    configurations.append({'loss_type': 'svm',
                           'l2': 0, 'epochs': 100, 'batch_size': 200, 'eta': 0.001, 'shuffle': True, 'decay': False})
    configurations.append({'loss_type': 'cross_entropy',
                           'l2': 0, 'epochs': 100, 'batch_size': 200, 'eta': 0.001, 'shuffle': True, 'decay': False})
    configurations.append({'loss_type': 'svm',
                           'l2': 0.001, 'epochs': 100, 'batch_size': 100, 'eta': 0.001, 'shuffle': True,
                           'decay': False})
    configurations.append({'loss_type': 'cross_entropy',
                           'l2': 0.0001, 'epochs': 100, 'batch_size': 50, 'eta': 0.0001, 'shuffle': True,
                           'decay': False})
    configurations.append({'loss_type': 'svm',
                           'l2': 0.0001, 'epochs': 100, 'batch_size': 30, 'eta': 0.0001, 'shuffle': True,
                           'decay': True})
    configurations.append({'loss_type': 'cross_entropy',
                           'l2': 0.001, 'epochs': 100, 'batch_size': 300, 'eta': 0.01, 'shuffle': True, 'decay': True})
    configurations.append({'loss_type': 'svm',
                           'l2': 0.001, 'epochs': 100, 'batch_size': 200, 'eta': 0.01, 'shuffle': True, 'decay': True})
    configurations.append({'loss_type': 'cross_entropy',
                           'l2': 0.001, 'epochs': 100, 'batch_size': 200, 'eta': 0.01, 'shuffle': True, 'decay': True})

    return configurations


def normalize_data(input, mu=None, sigma=None):
    if mu is None and sigma is None:
        mu = np.mean(input, axis=0).reshape(1, -1)

        sigma = np.std(input, axis=0).reshape(1, -1)

    normalized_data = (input - mu) / sigma
    return normalized_data, mu, sigma


def print_grad_diff(grad_w, grad_w_num):
    print('Grad W:')
    print('sum of abs difference: {}'.format(np.abs(grad_w - grad_w_num).sum()))
    print('mean of abs values W: {}   W_num: {}'
          .format(np.abs(grad_w).mean(), np.abs(grad_w_num).mean()))

    relative_error_w = np.abs(grad_w - grad_w_num) / np.maximum(np.abs(grad_w) + np.abs(grad_w_num),
                                                                1e-6 * np.ones(shape=grad_w.shape))

    # print('Relative error: {}'.format(relative_error_w))

    thresholds = [1e-05, 1e-06, 1e-07, 1e-08]
    for threshold in thresholds:
        low_enough = 0
        for i in range(relative_error_w.shape[0]):
            for j in range(relative_error_w.shape[1]):
                if relative_error_w[i][j] < threshold:
                    low_enough += 1

        percentage = round(low_enough * 100 / (relative_error_w.shape[0] * relative_error_w.shape[1]), 4)
        print("Low enough error is {0} for threshold {1}".format(percentage, threshold))


def compute_grads_for_matrix(y, x, W, model, grad_w):
    h = 1e-5
    grad_list = []
    for k in range(len(W)):

        grad_num = np.zeros_like(W[k])
        desc = 'Gradient computations for a {} layer weights, {} samples' \
            .format(k + 1, x.shape[0])
        with tqdm(desc=desc, total=W[k].size) as progress:

            for i in range(W[k].shape[0]):
                for j in range(W[k].shape[1]):
                    W_try = deepcopy(W[k])
                    W_try[i][j] -= h
                    model.layers[k * 2].W = W_try
                    model.forward_pass(x)
                    c1 = model.cost(y, None)

                    W_try = deepcopy(W[k])
                    W_try[i][j] += h
                    model.layers[k * 2].W = W_try
                    model.forward_pass(x)
                    c2 = model.cost(y, None)

                    grad_num[i][j] = (c2 - c1) / (2 * float(h))
                    progress.update()

            grad_list.append(grad_num)
            print('\n Layer {}'.format(k + 1))
            print_grad_diff(grad_w_num=grad_num, grad_w=grad_w[k])
    return grad_list


def to_one_hot(classification):
    hot_encoding = np.zeros([len(classification), np.max(classification) + 1])
    hot_encoding[np.arange(len(hot_encoding)), classification] = 1

    return hot_encoding


def load_cifar10(path=CIFAR_DIR, whole_dataset=False):
    def _load_batch(filename):
        with open(filename, 'rb') as f:
            datadict = pickle.load(f, encoding='latin1')
            X = datadict['data']
            Y = datadict['labels']
            X = X.reshape(10000, np.prod(X.shape[1:])).astype("float32") / 255

            Y = np.array(Y)
            return X, Y

    x_train = []
    y_train = []
    if whole_dataset:
        for batch in range(1, 6):
            print("=====>Loading Batch file: data_batch_{}<=====".format(batch))

            batch_filename = os.path.join(path, 'data_batch_{}'.format(batch))
            X, Y = _load_batch(batch_filename)
            x_train.append(X)
            y_train.append(Y)

        X_train = np.concatenate(x_train)
        y_train = np.concatenate(y_train)
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.025, random_state=1)

    else:
        X_train, y_train = _load_batch(os.path.join(path, 'data_batch_1'))
        X_val, y_val = _load_batch(os.path.join(path, 'data_batch_2'))
        X_test, y_test = _load_batch(os.path.join(path, 'test_batch'))

    y_train = to_one_hot(y_train)
    y_val = to_one_hot(y_val)
    y_test = to_one_hot(y_test)
    print("-----------------------------------------")
    print("           CIFAR10 is Loaded")
    print("-----------------------------------------")
    return X_train, y_train, X_val, y_val, X_test, y_test


def plot_loss_acc(train_loss, train_acc, val_loss, val_acc):
    fig = plt.figure()
    ax = fig.gca()
    plt.title('Final plot of cost for lambda: 0.006108, cycles: 3')
    plt.plot(train_loss, color='r')
    plt.plot(val_loss, color='g')
    plt.xlabel('Epochs')
    plt.legend(['train_cost', 'val_cost'])
    plt.grid()
    plt.show()

    fig = plt.figure()
    ax = fig.gca()
    plt.title('Final plot of accuracy for lambda: 0.006108, cycles: 3')
    plt.plot(train_acc, color='r')
    plt.plot(val_acc, color='g')
    plt.xlabel('Epochs')
    plt.legend(['train_accuracy', 'val_accuracy'])
    plt.grid()
    plt.show()


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

    def forward_pass(self, s):
        exp = np.exp(s.T)
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

        self.W = np.random.normal(0, self.std, size=(self.output_size, self.input_size))
        self.b = np.zeros((self.output_size, 1))
        # self.b = np.random.normal(0, self.std, size=(self.output_size, 1))

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
        self.grad_b = grad_b / N

        return np.dot(grad, self.W)

    def cost(self):
        powered_weights = np.power(self.W, 2).sum()
        J = np.multiply(self.l2, powered_weights)

        return J


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

        print("        Fit Started!!")
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

    noc = 3
    l_min = -5
    l_max = -1
    cyclical_values = {'eta_min': 1e-5, 'eta_max': 1e-1, 'noc': noc, 'k': 9}

    param = {}
    l = l_min + (l_max - l_min) * np.random.random_sample()
    l = np.power(10, l)

    param['l'] = round(l, 6)
    l = 0.006108
    start_time = time.time()

    dense1 = Dense(input_size=dim_size, output_size=50, l2_regul=l, std=1 / np.sqrt(dim_size))
    dense2 = Dense(input_size=50, output_size=10, l2_regul=l, std=1 / np.sqrt(50))

    model = Classifier()
    model.add_layer(dense1)
    model.add_layer(ReLU())
    model.add_layer(dense2)
    model.add_layer(SoftMax())

    train_loss, val_loss, train_acc, val_acc = model.fit(X_train,
                                                         y_train,
                                                         X_val,
                                                         y_val,
                                                         cyclical_values=cyclical_values,
                                                         batch_size=100)

    accuracy = model.predict(X_test, y_test)
    end_time = time.time() - start_time

    param['accuracy on test'] = accuracy
    param['accuracy on validation'] = np.amax(val_acc)

    param['number of cycles'] = noc
    param['time'] = round(end_time, 2)

    Utils.write_to_csv(param)

    Utils.plot_loss_acc(train_loss, train_acc, val_loss, val_acc)
