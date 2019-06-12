import pickle
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import re


def print_grad_diff(grad, grad_num):
    print('sum of abs difference: {}'.format(np.abs(grad - grad_num).sum()))
    print('mean of abs values: {}   num: {}'
          .format(np.abs(grad).mean(), np.abs(grad_num).mean()))

    relative_error = np.abs(grad - grad_num) / np.maximum(np.abs(grad) + np.abs(grad_num),
                                                          1e-8 * np.ones(shape=grad.shape))

    print("Max relative error {}".format(np.max(relative_error)))
    thresholds = [1e-05, 1e-06, 1e-07, 1e-08]
    for threshold in thresholds:
        low_enough = 0
        for i in range(relative_error.shape[0]):
            for j in range(relative_error.shape[1]):
                if relative_error[i][j] < threshold:
                    low_enough += 1

        percentage = round(low_enough * 100 / (relative_error.shape[0] * relative_error.shape[1]), 4)
        print("Low enough error is {0} for threshold {1}".format(percentage, threshold))


def compute_grads_num_slow(X, Y, rnn, h_t, h=1e-5):
    # For C
    print('Grad c :')

    rnn_grad = rnn.d_c
    c = rnn.c
    grad_c = np.zeros(c.shape)

    for i in range(c.shape[0]):
        c_try = deepcopy(c)
        c_try[i] -= h

        rnn.c = c_try
        _, _, c1 = rnn.forward(X, Y, h_t)

        c_try = deepcopy(c)
        c_try[i] += h

        rnn.c = c_try
        _, _, c2 = rnn.forward(X, Y, h_t)

        grad_c[i] = (c2 - c1) / (2 * float(h))

    print_grad_diff(grad_num=grad_c, grad=rnn_grad.reshape(-1, 1))

    rnn.c = c
    # For b
    print('Grad b :')

    rnn_grad = rnn.d_b
    b = rnn.b
    grad_b = np.zeros(b.shape)

    for i in range(b.shape[0]):
        b_try = deepcopy(b)
        b_try[i] -= h

        rnn.b = b_try
        _, _, c1 = rnn.forward(X, Y, h_t)

        b_try = deepcopy(b)
        b_try[i] += h

        rnn.b = b_try
        _, _, c2 = rnn.forward(X, Y, h_t)

        grad_b[i] = (c2 - c1) / (2 * float(h))
    print_grad_diff(grad_num=grad_b, grad=rnn_grad.reshape(-1, 1))

    rnn.b = b
    # For W
    print('Grad W :')

    rnn_grad = rnn.d_W
    W = rnn.W
    grad_W = np.zeros(W.shape)

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            matrix_try = deepcopy(W)
            matrix_try[i][j] -= h

            rnn.W = matrix_try
            _, _, c1 = rnn.forward(X, Y, h_t)

            matrix_try = deepcopy(W)
            matrix_try[i][j] += h

            rnn.W = matrix_try
            _, _, c2 = rnn.forward(X, Y, h_t)

            grad_W[i][j] = (c2 - c1) / (2 * float(h))

    print_grad_diff(grad_num=grad_W, grad=rnn_grad)

    rnn.W = W
    # For U
    print('Grad U :')

    rnn_grad = rnn.d_U
    U = rnn.U
    grad_U = np.zeros(U.shape)

    for i in range(U.shape[0]):
        for j in range(U.shape[1]):
            matrix_try = deepcopy(U)
            matrix_try[i][j] -= h

            rnn.U = matrix_try
            _, _, c1 = rnn.forward(X, Y, h_t)

            matrix_try = deepcopy(U)
            matrix_try[i][j] += h

            rnn.U = matrix_try
            _, _, c2 = rnn.forward(X, Y, h_t)

            grad_U[i][j] = (c2 - c1) / (2 * float(h))

    print_grad_diff(grad_num=grad_U, grad=rnn_grad)

    rnn.U = U
    # For V
    print('Grad V :')

    rnn_grad = rnn.d_V
    V = rnn.V
    grad_V = np.zeros(V.shape)

    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            matrix_try = deepcopy(V)
            matrix_try[i][j] -= h

            rnn.V = matrix_try
            _, _, c1 = rnn.forward(X, Y, h_t)

            matrix_try = deepcopy(V)
            matrix_try[i][j] += h

            rnn.V = matrix_try
            _, _, c2 = rnn.forward(X, Y, h_t)

            grad_V[i][j] = (c2 - c1) / (2 * float(h))

    print_grad_diff(grad_num=grad_V, grad=rnn_grad)

def plot_loss(loss):
    fig = plt.figure()
    ax = fig.gca()
    plt.title('Evolution of the smooth loss by RNN')
    plt.plot(loss, color='g')
    plt.xlabel('Update Steps')
    plt.legend(['smooth_loss'])
    plt.grid()
    plt.show()

def cross_entropy_loss(Y, P):
    loss = - np.log(P[Y == 1])
    return loss.sum()


def softmax(input):
    exp = np.exp(input)
    sum = np.sum(exp, axis=0)
    p = (exp / sum)
    return p


def to_one_hot(input, char_to_ind):
    hot_encoding = np.zeros((len(char_to_ind), len(input)))

    for i, x in enumerate(input):
        hot_encoding[char_to_ind[x], i] = 1

    return hot_encoding


def to_text(input, ind_to_char):
    text = ''

    for i, x in enumerate(input.T):
        idx = np.where(x == 1)[0]
        text += ind_to_char[int(idx)]
    return text


def load_book():
    book = open('goblet_book.txt', encoding='utf-8').read()
    book_chars = list({char for char in book})

    char_to_ind = {book_chars[i]: i for i in range(len(book_chars))}
    ind_to_char = {ind: char for char, ind in char_to_ind.items()}

    charset = set(open('goblet_book.txt').read())
    chars = ''.join(charset)
    filtered_chars = re.sub('[+]', '', chars)

    return book, char_to_ind, ind_to_char



def save_model_params(state):
    with open('best_model', 'wb') as handle:
        pickle.dump(state, handle)

def load_model_params():
    with open('best_model', 'rb') as handle:
        b = pickle.load(handle)
    return b

if __name__ == "__main__":
    book_chars, char_to_ind, ind_to_char = load_book()

    m = 100
    eta = .1
    seq_length = 25
