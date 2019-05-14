import time
import Utils
import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from sklearn.model_selection import train_test_split

import csv
import operator

np.random.seed(12345)


class Classifier(object):
    def __init__(self, train_data, val_data, val_labels, train_labels,
                 loss_type='cross_entropy',
                 l2=0.01,
                 eta=0.01,
                 shuffled=False,
                 decay=False,
                 xavier_init=False):

        self.train_data = train_data
        self.val_data = val_data
        self.val_labels = val_labels
        self.train_labels = train_labels
        self.xavier_init = xavier_init
        self.something = 0
        self.loss_type = loss_type
        self.weights, self.bias = self._initialize_weights(train_labels.shape[1], train_data.shape[1])
        self.shuffled = shuffled
        self.decay = decay
        self.val_loss = []
        self.train_loss = []
        self.val_acc = []
        self.train_acc = []
        self.eta = eta
        self.l2 = l2

        if self.loss_type is 'cross_entropy':
            self.compute_loss = self.cross_entropy_loss
            self.backward_pass = self.compute_gradients_cross_entropy
            self.predict = self.predict_cross_entropy
        elif self.loss_type is 'svm':
            self.compute_loss = self.svm_loss
            self.backward_pass = self.compute_gradients_svm
            self.predict = self.predict_svm
        else:
            print("No implemented loss has been defined.")
            exit()

    def _initialize_weights(self, K, d):
        if self.xavier_init:
            weights = np.random.normal(0, 1 / np.sqrt(d), size=(K, d))
            bias = np.random.normal(0, 1 / np.sqrt(d), size=(K, 1))
        else:
            weights = np.random.normal(0, 0.01, size=(K, d))
            bias = np.random.normal(0, 0.01, size=(K, 1))
        return weights, bias

    def update_weights(self, grad_w, grad_b):
        self.weights -= np.multiply(self.eta, grad_w)
        self.bias -= np.multiply(self.eta, grad_b)

    def fit(self, batch_size=100, n_epochs=1):
        N = self.train_data.shape[0]
        n_batch_loops = int(N / batch_size)
        indexes = np.arange(N)
        for i in (range(n_epochs)):

            if self.shuffled:
                np.random.shuffle(indexes)

            for j in range(1, n_batch_loops):
                j_start = (j - 1) * batch_size
                j_end = j * batch_size
                batch_indexes = indexes[j_start: j_end]

                x_batch = self.train_data[batch_indexes]
                y_batch = self.train_labels[batch_indexes]

                if self.loss_type is 'cross_entropy':
                    fwd = self.forward_pass(x_batch)
                else:
                    fwd = []

                grad_w, grad_b = self.backward_pass(fwd, x_batch, y_batch)

                self.update_weights(grad_w, grad_b)

            val_loss = self.compute_loss(self.val_data, self.val_labels)
            train_loss = self.compute_loss(self.train_data, self.train_labels)
            _, val_acc = self.predict(self.val_data, self.val_labels)
            _, train_acc = self.predict(self.train_data, self.train_labels)

            self.val_loss.append(val_loss)
            self.train_loss.append(train_loss)
            self.val_acc.append(val_acc)
            self.train_acc.append(train_acc)

            print("Epoch={0}, Val_acc={1}%, Val_Loss={2}, Train_acc={3}%, Train_Loss={4}".format(
                i + 1, val_acc, round(val_loss, 4), train_acc, round(train_loss, 4)))
            if self.decay:
                self.eta *= 0.9
        return np.array(self.train_loss), np.array(self.val_loss), np.array(self.train_acc), np.array(self.val_acc)

    def forward_pass(self, x):
        product = np.dot(self.weights, x.T)
        sum = product + self.bias
        output = Utils.softmax(sum)

        return output

    def compute_gradients_cross_entropy(self, fwd, x_batch, y_batch):
        N = x_batch.shape[0]
        fwd = fwd.T

        grad_w = np.zeros_like(self.weights)
        grad_b = np.zeros_like(self.bias)

        for i in range(N):
            x = x_batch[i, :]
            y = y_batch[i, :]
            p = fwd[i, :]

            g = p - y

            grad_w += np.outer(g, x)
            grad_b += np.reshape(g, grad_b.shape)

        # grad_w = np.mean(grad_w) + 2*np.multiply(self.l2, self.weights)
        grad_w = (grad_w / N) + 2 * np.multiply(self.l2, self.weights)
        grad_b /= N

        return grad_w, grad_b

    def compute_gradients_svm(self, fwd, x_batch, y_batch):
        N = x_batch.shape[0]

        product = np.dot(self.weights, x_batch.T)
        s_j = (product + self.bias).T

        indices = np.where(y_batch == 1)
        s_y = s_j[indices]

        s = s_j - s_y.reshape(-1, 1) + 1

        indicators = np.where(s > 0, 1, 0)

        sums = np.sum(np.where(y_batch.T == 1, 0, indicators.T), axis=0)

        g = np.where(y_batch.T == 1, -sums, indicators.T)

        grad_w = np.dot(g, x_batch)

        grad_w = (grad_w / N) + 2 * self.l2 * self.weights
        grad_b = (g.sum(axis=1).reshape(-1, 1) / N)

        return grad_w, grad_b

    def cross_entropy_loss(self, x_batch, y_batch):
        p = self.forward_pass(x_batch)

        l_cross = np.multiply(y_batch.T, p).sum(axis=0)

        J = - np.mean(np.log(l_cross)).sum()

        weights = np.power(self.weights, 2).sum()
        J += np.multiply(self.l2, weights)

        return J

    def svm_loss(self, x_batch, y_batch):
        product = np.dot(self.weights, x_batch.T)
        s_j = (product + self.bias).T

        indices = np.where(y_batch == 1)

        s_y = s_j[indices]

        base = s_j - s_y.reshape(-1, 1) + 1
        base = base.T

        loss = np.sum(np.where((y_batch.T == 1) | (base < 0), 0, base))

        hinge_loss = loss / x_batch.shape[0]
        regularisation_loss = (self.l2) * np.power(self.weights, 2).sum()

        loss = regularisation_loss + hinge_loss

        return loss

    def predict_cross_entropy(self, x, y):
        fwd = self.forward_pass(x)
        pred = np.argmax(fwd, axis=0)

        true_indices = np.where(y == 1)[1]
        correct_pred = np.where(true_indices == pred)[0]

        accuracy = correct_pred.shape[0] * 100 / pred.shape[0]
        round_accuracy = round(accuracy, 3)

        return pred, round_accuracy

    def predict_svm(self, x, y):

        product = np.dot(self.weights, x.T)
        pred = np.argmax(product, axis=0)

        true_indices = np.where(y == 1)[1]
        correct_pred = np.where(true_indices == pred)[0]

        accuracy = correct_pred.shape[0] * 100 / pred.shape[0]
        round_accuracy = round(accuracy, 3)

        return pred, round_accuracy


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # Directory of the script
CIFAR_DIR = os.path.join(ROOT_DIR, 'CIFAR10/')  # Directory of the dataset
PLOT_DIR = os.path.join(ROOT_DIR, 'plots/')


def write_to_csv(dictionary):
    with open('thresholds_grad.csv', 'a', newline='') as csvfile:
        fieldnames = dictionary.keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(dictionary)


def print_grad_diff(grad_w, grad_w_num, grad_b, grad_b_num):
    print('Grad W:')
    print('sum of abs difference: {}'.format(np.abs(grad_w - grad_w_num).sum()))
    print('mean of abs values W: {}   W_num: {}'
          .format(np.abs(grad_w).mean(), np.abs(grad_w_num).mean()))

    print('Grad b:')
    print('sum of abs differences: {}'.format(np.abs(grad_b - grad_b_num).sum()))
    print('mean of abs values b: {}   b_num: {}'
          .format(np.abs(grad_b).mean(), np.abs(grad_b_num).mean()))

    relative_error_w = np.abs(grad_w - grad_w_num) / np.maximum(np.abs(grad_w) + np.abs(grad_w_num),
                                                                1e-6 * np.ones(shape=grad_w.shape))
    relative_error_b = np.abs(grad_b - grad_b) / np.maximum(np.abs(grad_b) + np.abs(grad_b),
                                                            1e-6 * np.ones(shape=grad_b.shape))

    print('Relative error: {}'.format(relative_error_w))

    for threshold in [1e-5, 1e-6, 1e-7, 1e-8]:
        low_enough = 0
        for i in range(relative_error_w.shape[0]):
            for j in range(relative_error_w.shape[1]):
                if relative_error_w[i][j] < threshold:
                    low_enough += 1

        percentage = round(low_enough * 100 / (relative_error_w.shape[0] + relative_error_w.shape[1]), 4)
        print("Low enough error is {} for threshold {}", percentage, threshold)


def to_one_hot(classification):
    hot_encoding = np.zeros([len(classification), np.max(classification) + 1])
    hot_encoding[np.arange(len(hot_encoding)), classification] = 1

    return hot_encoding


def load_cifar10(path=CIFAR_DIR):
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
    # X_train, y_train = _load_batch(os.path.join(path, 'data_batch_1'))
    # X_val, y_val = _load_batch(os.path.join(path, 'data_batch_2'))
    # X_test, y_test = _load_batch(os.path.join(path, 'test_batch'))

    print("-----------------------------------------")
    print("           CIFAR10 is Loaded")
    print("-----------------------------------------")
    return X_train, y_train, X_val, y_val, X_test, y_test


def softmax(s):
    exp = np.exp(s)
    return exp / np.sum(exp, axis=0)


def plot_image(image):
    """Expects 1x3072 image size"""
    plt.imshow(image.reshape(3, 32, 32).transpose(1, 2, 0))
    plt.show()


def plot_errors(train_loss, val_loss, values, legends, to_be_title):
    title = ''
    for key, value in to_be_title.items():
        title += key + ': ' + str(value) + ' '
    title = title[11:-29]
    plt.figure()
    plt.title(title)
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.legend(legends)
    plt.xlabel('Epochs')
    plt.grid()
    plt.savefig(PLOT_DIR + 'val_acc/errors_{}.png'.format(values))


def compute_grads_num_slow(X, Y, W, b, lamda=0.001, h=1e-6):
    def softmax(s):
        exp_s = np.exp(s)
        return exp_s / exp_s.sum(axis=0)

    def evaluate_classifier(X, W, b):
        S = np.dot(W, X) + b
        P = softmax(S)
        return P

    def cross_entropy_loss(X, Y, W, b_try, lamda):
        P = evaluate_classifier(X, W, b_try)

        l_cross = np.multiply(Y.T, P).sum(axis=0)

        # l_cross[l_cross == 0] = np.finfo(float).eps

        J = - np.mean(np.log(l_cross)).sum()

        weights = np.power(W, 2).sum()

        J += np.multiply(lamda, weights)

        return J

    grad_W = np.zeros(W.shape)
    grad_b = np.zeros(b.shape)

    for i in range(b.shape[0]):
        b_try = deepcopy(b)
        b_try[i] -= h
        c1 = cross_entropy_loss(X, Y, W, b_try, lamda)
        b_try = deepcopy(b)
        b_try[i] += h
        c2 = cross_entropy_loss(X, Y, W, b_try, lamda)
        grad_b[i] = (c2 - c1) / (2 * float(h))

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W_try = deepcopy(W)
            W_try[i][j] -= h
            c1 = cross_entropy_loss(X, Y, W_try, b, lamda)
            W_try = deepcopy(W)
            W_try[i][j] += h
            c2 = cross_entropy_loss(X, Y, W_try, b, lamda)
            grad_W[i][j] = (c2 - c1) / (2 * float(h))

    return grad_W, grad_b


def visualize_weights(weights, values):
    for i, row in enumerate(weights):
        plt.subplot(1, 10, i + 1)
        frame1 = plt.gca()
        frame1.axes.xaxis.set_ticklabels([])
        frame1.axes.yaxis.set_ticklabels([])
        img = (row - row.min()) / (row.max() - row.min())
        image = np.reshape(img, (32, 32, 3), order='F')
        plt.imshow(image)
        plt.title('Weight {}'.format(i + 1))
    plt.show()
    # plt.savefig(PLOT_DIR + 'weights/weights_{}.png'.format(values))


def sort_csv():
    reader = csv.reader(open("thresholds_grad.csv"), delimiter=",")
    sortedlist = sorted(reader, key=operator.itemgetter(8), reverse=False)
    with open('sorted.csv', 'w') as f:
        fieldnames = ['loss_type', 'l2,epochs', 'batch_size', 'eta', 'shuffle', 'decay', 'time', 'accuracy']
        writer = csv.writer(f)
        # writer.writeheader()
        for row in sortedlist:
            writer.writerow(row)


if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test = Utils.load_cifar10()

    y_train = Utils.to_one_hot(y_train)
    y_val = Utils.to_one_hot(y_val)
    y_test = Utils.to_one_hot(y_test)

    configurations = []
    configurations.append({'loss_type': 'cross_entropy',
                           'l2': 0.0001, 'epochs': 2, 'batch_size': 100, 'eta': 0.1, 'shuffle': False, 'decay': False})
    # configurations.append({'loss_type': 'cross_entropy',
    #                        'l2': 0., 'epochs': 40, 'batch_size': 100, 'eta': 0.01, 'shuffle': False, 'decay': False})
    # configurations.append({'loss_type': 'cross_entropy',
    #                        'l2': 0.1, 'epochs': 40, 'batch_size': 100, 'eta': 0.01, 'shuffle': False, 'decay': False})
    # configurations.append({'loss_type': 'cross_entropy',
    #                        'l2': 1, 'epochs': 40, 'batch_size': 100, 'eta': 0.01, 'shuffle': False, 'decay': False})
    # configurations.append({'loss_type': 'svm',
    #                        'l2': 0.01, 'epochs': 50, 'batch_size': 100, 'eta': 0.01, 'shuffle': False, 'decay': False})
    # configurations.append({'loss_type': 'cross_entropy',
    #                        'l2': 0.01, 'epochs': 50, 'batch_size': 100, 'eta': 0.01, 'shuffle': False, 'decay': False})
    # configurations.append({'loss_type': 'svm',
    #                        'l2': 1, 'epochs': 50, 'batch_size': 100, 'eta': 0.1, 'shuffle': False, 'decay': False})
    # configurations.append({'loss_type': 'cross_entropy',
    #                        'l2': 1, 'epochs': 50, 'batch_size': 100, 'eta': 0.1, 'shuffle': False, 'decay': False})
    # configurations.append({'loss_type': 'svm',
    #                        'l2': 0, 'epochs': 100, 'batch_size': 200, 'eta': 0.001, 'shuffle': True, 'decay': False})
    # configurations.append({'loss_type': 'cross_entropy',
    #                        'l2': 0, 'epochs': 100, 'batch_size': 200, 'eta': 0.001, 'shuffle': True, 'decay': False})
    # configurations.append({'loss_type': 'svm',
    #                        'l2': 0.001, 'epochs': 100, 'batch_size': 100, 'eta': 0.001, 'shuffle': True,
    #                        'decay': False})
    # configurations.append({'loss_type': 'cross_entropy',
    #                        'l2': 0.0001, 'epochs': 100, 'batch_size': 50, 'eta': 0.0001, 'shuffle': True,
    #                        'decay': False})
    # configurations.append({'loss_type': 'svm',
    #                        'l2': 0.0001, 'epochs': 100, 'batch_size': 30, 'eta': 0.0001, 'shuffle': True,
    #                        'decay': True})
    # configurations.append({'loss_type': 'cross_entropy',
    #                        'l2': 0.001, 'epochs': 100, 'batch_size': 300, 'eta': 0.01, 'shuffle': True, 'decay': True})
    # configurations.append({'loss_type': 'svm',
    #                        'l2': 0.001, 'epochs': 100, 'batch_size': 200, 'eta': 0.01, 'shuffle': True, 'decay': True})
    # configurations.append({'loss_type': 'cross_entropy',
    #                        'l2': 0.001, 'epochs': 100, 'batch_size': 200, 'eta': 0.01, 'shuffle': True, 'decay': True})

    for i, values in enumerate((configurations)):
        print("===================> Start of Training of case {0} <===================".format(i + 1))
        start = time.time()
        model = Classifier(X_train, X_val, y_val, y_train,
                           loss_type=values['loss_type'],
                           l2=values['l2'],
                           eta=values['eta'],
                           shuffled=values['shuffle'],
                           decay=values['decay'])

        train_loss, val_loss, train_acc, val_acc = model.fit(
            n_epochs=values['epochs'],
            batch_size=values['batch_size'])

        fwd = model.forward_pass(X_train)

        model_gad_w, model_grad_b = model.compute_gradients_cross_entropy(fwd, X_train[:1], y_train[:1])
        num_grad_w, num_grad_b = compute_grads_num_slow(X_train[:1].T, y_train[:1], model.weights, model.bias)

        print_grad_diff(model_gad_w, num_grad_w, model_grad_b, num_grad_b)

        # _, accuracy = model.predict(X_test, y_test)
        #
        # print("Accuracy on test set is : {}%".format(accuracy))
        #
        # Utils.plot_errors(train_loss, val_loss, 'loss_try{}'.format(i + 1),
        #                   legends=['train_loss', 'val_loss'],
        #                   to_be_title=values)
        # Utils.plot_errors(train_acc, val_acc, 'acc_try{}'.format(i + 1),
        #                   legends=['train_acc', 'val_acc'],
        #                   to_be_title=values)
        # Utils.visualize_weights(model.weights, 'try_{}'.format(i + 1))
        #
        # end = time.time() - start
        #
        # values['time'] = round(end, 3)
        # values['accuracy'] = accuracy
        #
        # write_to_csv(values)
        #
        # print("Time for {0} test case was: {1} sec.".format(i + 1, round(end, 3)))
