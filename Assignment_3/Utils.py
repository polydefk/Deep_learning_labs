import csv
import pickle
import os
from sklearn.model_selection import train_test_split
from Assignment_3.model import *

from tqdm import tqdm
from copy import deepcopy
import numpy as np

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # Directory of the script
CIFAR_DIR = os.path.join(ROOT_DIR, 'CIFAR10/')  # Directory of the dataset
PLOT_DIR = os.path.join(ROOT_DIR, 'plots/')


def plot_loss_acc(train_loss, train_acc, val_loss, val_acc):
    fig = plt.figure()
    ax = fig.gca()
    plt.title('Final plot of cost')
    plt.plot(train_loss, color='r')
    plt.plot(val_loss, color='g')
    plt.xlabel('Epochs')
    plt.legend(['train_cost', 'val_cost'])
    plt.grid()

    fig = plt.figure()
    ax = fig.gca()
    plt.title('Final plot of accuracy')
    plt.plot(train_acc, color='r')
    plt.plot(val_acc, color='g')
    plt.xlabel('Epochs')
    plt.legend(['train_accuracy', 'val_accuracy'])
    plt.grid()
    plt.show()


def write_to_csv(dictionary):
    with open('results.csv', 'a', newline='') as csvfile:
        fieldnames = dictionary.keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        # writer.writeheader()
        writer.writerow(dictionary)


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
    print('mean of abs values: {}   W_num: {}'
          .format(np.abs(grad_w).mean(), np.abs(grad_w_num).mean()))

    relative_error = np.abs(grad_w - grad_w_num) / np.maximum(np.abs(grad_w) + np.abs(grad_w_num),
                                                              1e-6 * np.ones(shape=grad_w.shape))

    # print('Relative error: {}'.format(relative_error))
    #
    # max_err = relative_error.max()
    # n_ok = (relative_error < 1e-05).sum()
    # p_ok = n_ok / grad_w_num.size * 100
    #
    # print(f'Max error: {max_err}\nPercentage of values under max tolerated value: {p_ok}\n' +
    #       f'eps: {1e-16}\tMax tolerated error: {1e-05}')

    low_enough = 0
    for i in range(relative_error.shape[0]):
        for j in range(relative_error.shape[1]):
            if relative_error[i][j] < 1e-05:
                low_enough += 1

    percentage = round(low_enough * 100 / (relative_error.shape[0] * relative_error.shape[1]), 4)
    print("Low enough error is {0} for threshold {1}".format(percentage, 1e-5))



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
                    model.layers[k * 3].W = W_try
                    model.forward_pass(x)
                    c1 = model.cost(y, None)

                    W_try = deepcopy(W[k])
                    W_try[i][j] += h
                    model.layers[k * 3].W = W_try
                    model.forward_pass(x)
                    c2 = model.cost(y, None)

                    grad_num[i][j] = (c2 - c1) / (2 * float(h))
                    progress.update()

            grad_list.append(grad_num)
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

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.125, random_state=1)

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


if __name__ == '__main__':
    X_train, y_train, X_val, y_val, X_test, y_test = load_cifar10()
