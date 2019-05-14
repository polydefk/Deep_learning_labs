import csv
import pickle
import os
from sklearn.model_selection import train_test_split
from model import *
from tqdm import tqdm
from copy import deepcopy
import numpy as np

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # Directory of the script
CIFAR_DIR = os.path.join(ROOT_DIR, 'CIFAR10/')  # Directory of the dataset
PLOT_DIR = os.path.join(ROOT_DIR, 'plots/')


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


def overfiting_without_regul():
    X_train, y_train, X_val, y_val, X_test, y_test = Utils.load_cifar10(whole_dataset=False)
    X_train, mu, sigma = Utils.normalize_data(X_train)
    X_val, *_ = Utils.normalize_data(X_val, mu, sigma)
    X_test, *_ = Utils.normalize_data(X_test, mu, sigma)

    N = X_train.shape[0]
    d = X_train.shape[1]
    train_examples = 100
    dim_size = d

    X_train = X_train[:train_examples, :dim_size]
    y_train = y_train[:train_examples, :dim_size]
    X_val = X_val[:train_examples, :dim_size]
    y_val = y_val[:train_examples, :dim_size]

    dense1 = Dense(input_size=X_train.shape[1], output_size=50, l2_regul=0, std=1 / np.sqrt(dim_size))
    dense2 = Dense(input_size=50, output_size=10, l2_regul=0, std=1 / np.sqrt(50))

    model = Classifier()
    model.add_layer(dense1)
    model.add_layer(ReLU())
    model.add_layer(dense2)
    model.add_layer(SoftMax())

    train_loss, val_loss, train_acc, val_acc = model.fit(X_train, y_train, X_val, y_val,
                                                         eta=0.001,
                                                         batch_size=10,
                                                         n_epochs=200,
                                                         cyclical_values={})

    plot_loss_acc(train_loss, train_acc, val_loss, val_acc)


def test_cyclical_learning():
    X_train, y_train, X_val, y_val, X_test, y_test = Utils.load_cifar10(whole_dataset=False)

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

    dense1 = Dense(input_size=dim_size, output_size=50, l2_regul=0.01, std=1 / np.sqrt(dim_size))
    dense2 = Dense(input_size=50, output_size=10, l2_regul=0.01, std=1 / np.sqrt(50))

    model = Classifier()
    model.add_layer(dense1)
    model.add_layer(ReLU())
    model.add_layer(dense2)
    model.add_layer(SoftMax())

    cyclical_values = {'eta_min': 1e-5, 'eta_max': 1e-1, 'noc': 3, 'k': 8}

    train_loss, val_loss, train_acc, val_acc = model.fit(X_train, y_train, X_val, y_val,
                                                         cyclical_values=cyclical_values)

    accuracy = model.predict(X_test, y_test)

    plot_loss_acc(train_loss, train_acc, val_loss, val_acc)
    print("Final accuracy is {}".format(accuracy))


if __name__ == '__main__':
    test_cyclical_learning()
