from Assignment_4.model import *


def check_gradients():
    book, char_to_ind, ind_to_char = load_book()

    K = len(char_to_ind)
    m = 5
    n = 25

    x = to_one_hot(book[0:n], char_to_ind)
    y = to_one_hot(book[1:n + 1], char_to_ind)
    h_t = np.zeros((m, 1))

    rnn = Vanila_RNN(m=m, K=K)
    rnn.forward(x, y, h_t)
    rnn.backward(y_true=y)

    compute_grads_num_slow(x, y, rnn, h_t)


if __name__ == "__main__":
    check_gradients()