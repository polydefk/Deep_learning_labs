import pickle
import os
import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # Directory of the script
CIFAR_DIR = os.path.join(ROOT_DIR, 'CIFAR10/')  # Directory of the dataset


def to_one_hot(classification):
    hot_encoding = np.zeros([len(classification), np.max(classification) + 1])
    hot_encoding[np.arange(len(hot_encoding)), classification] = 1

    return hot_encoding


def _load_batch(filename):
    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding='latin1')
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, np.prod(X.shape[1:])).astype("float32") / 255

        Y = np.array(Y)
        return X, Y


def load_cifar10(path=CIFAR_DIR):
    # x_train = []
    # y_train = []
    # for batch in range(1, 6):
    #     print("=====>Loading Batch file: data_batch_{}<=====".format(batch))
    #
    #     batch_filename = os.path.join(path, 'data_batch_{}'.format(batch))
    #     # print(batch_filename)
    #     X, Y = _load_batch(batch_filename)
    #     x_train.append(X)
    #     y_train.append(Y)
    #
    # X_train = np.concatenate(x_train)
    # y_train = np.concatenate(y_train)

    X_train, y_train = _load_batch(os.path.join(path, 'data_batch_1'))
    X_val, y_val = _load_batch(os.path.join(path, 'data_batch_2'))
    X_test, y_test = _load_batch(os.path.join(path, 'test_batch'))

    print("-----------------------------------------")
    print("           CIFAR10 is Loaded")
    print("-----------------------------------------")

    return X_train, y_train, X_val, y_val, X_test, y_test


def softmax(s):
    return np.exp(s) / np.sum(np.exp(s), axis=0)


def plot_image(image):
    """Expects 1x3072 image size"""
    plt.imshow(image.reshape(3, 32, 32).transpose(1, 2, 0))
    plt.show()


if __name__ == "__main__":
    # s = np.array([1, 2, 3, 4, 5])
    # print (softmax(s))
    # a, b, c, d, e, f = load_CIFAR10(CIFAR_DIR)
    #
    # model = Classifier(a, to_One_Hot(b))
    # s = np.dot(model.weights.T, a) + model.bias
    pass