import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from sklearn.model_selection import train_test_split
import Assignment_1.Model as Model
import csv
import operator

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
    print(np.abs(grad_w - grad_w_num).sum())
    print(np.abs(grad_w).mean(), np.abs(grad_w_num).mean())

    print('Grad b:')
    print(np.abs(grad_b - grad_b_num).sum())
    print(np.abs(grad_b).mean(), np.abs(grad_b_num).mean())


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
    print(y_val.shape)
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
    # plt.show()
    plt.savefig(PLOT_DIR + 'val_acc/errors_{}.png'.format(values))


def compute_grads_num_slow(X, Y, W, b, lamda=0.001, h=1e-6):
    def softmax(s):
        exp_s = np.exp(s)
        return exp_s / exp_s.sum(axis=0)

    def evaluate_classifier(X, W, b):
        S = np.dot(W, X) + b
        P = softmax(S)
        return P

    def l_cross_entropy(Y, P):
        loss = - np.log(P.T[Y.T == 1].T)
        return loss

    def compute_cost(X, Y, W, b, lamda):
        P = evaluate_classifier(X, W, b)
        J = l_cross_entropy(Y, P).sum() / float(X.shape[1])
        J += lamda * np.square(W).sum()
        return J

    grad_W = np.zeros(W.shape)
    grad_b = np.zeros(b.shape)

    for i in range(b.shape[0]):
        b_try = deepcopy(b)
        b_try[i] -= h
        c1 = compute_cost(X, Y, W, b_try, lamda)
        b_try = deepcopy(b)
        b_try[i] += h
        c2 = compute_cost(X, Y, W, b_try, lamda)
        grad_b[i] = (c2 - c1) / (2 * float(h))

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W_try = deepcopy(W)
            W_try[i][j] -= h
            c1 = compute_cost(X, Y, W_try, b, lamda)
            W_try = deepcopy(W)
            W_try[i][j] += h
            c2 = compute_cost(X, Y, W_try, b, lamda)
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
        plt.title('Weight {}'.format(i+1))
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


if __name__ == '__main__':
    X_train, y_train, X_val, y_val, X_test, y_test = load_cifar10()

    y_train = to_one_hot(y_train)
    y_val = to_one_hot(y_val)
    y_test = to_one_hot(y_test)

    model = Model.Classifier(X_train, X_val, y_val, y_train, loss_type='svm')
    # train_loss, val_loss, train_acc, val_acc = model.fit()
    X_test = X_test[:20]
    y_test = y_test[:20]

    p = model.forward_pass(X_test)
    grad_w, grad_b = model.backward_pass(p, X_test, y_test)

    grad_w1, grad_b1 = compute_grads_num_slow(X_test.T, y_test.T, model.weights, model.bias)

    print_grad_diff(grad_w, grad_w1, grad_b, grad_b1)

    values = {'l2': 0, 'epochs': 100, 'batch_size': 200, 'eta': 0.1, 'shuffle': False, 'decay': False}
    # plot_errors(train_acc, val_acc, 'yolo{}'.format(1), legends=['train_acc', 'val_acc'], to_be_title=values)

