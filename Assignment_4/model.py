from Utils import *
from Assignment_4.Utils import *


class Vanila_RNN():
    def __init__(self, m, K):
        self.m = m
        self.K = K

        self.b = np.zeros((m, 1))
        self.c = np.zeros((K, 1))

        self.W = np.random.normal(0, 0.01, size=(m, m))
        self.U = np.random.normal(0, 0.01, size=(m, K))
        self.V = np.random.normal(0, 0.01, size=(K, m))

        self.epsilon = 1e-8
        self.eta = 0.01

    def predict_char(self, h_0, x_1):
        a_t = np.dot(self.W, h_0) + np.dot(self.U, x_1).reshape(-1, 1) + self.b  # m,1 size

        h_t = np.tanh(a_t)

        o_t = np.dot(self.V, h_t) + self.c  # K,1 size

        p_t = softmax(o_t).flatten()

        idx = np.random.choice(self.K, 1, p=p_t)

        return idx, h_t, p_t, a_t

    def synthesize(self, h_t, x, ind_to_char, seq_length=200):

        prediction = np.zeros((self.K, seq_length))
        pred = x
        for i in range(seq_length):
            idx, h_t, _, _ = self.predict_char(h_t, pred)
            prediction[idx, i] = 1
            pred = prediction[:, i]

        return to_text(prediction, ind_to_char)

    def forward(self, x, y_true, h_t):
        n = x.shape[1]

        self.X = x
        self.y_pred = np.zeros((self.K, n))
        self.a_t = np.zeros((self.m, n))
        self.h_t = np.zeros((self.m, n))
        self.p_t = np.zeros((self.K, n))

        for i in range(n):
            idx, h_t, p_t, a_t = self.predict_char(h_t, x[:, i])
            self.y_pred[idx, i] = 1
            self.a_t[:, i] = a_t.flatten()
            self.h_t[:, i] = h_t.flatten()
            self.p_t[:, i] = p_t.flatten()

        loss = cross_entropy_loss(y_true, self.p_t)

        return self.y_pred, self.p_t, loss

    def backward(self, y_true):
        n = self.X.shape[1]

        d_o_t = self.p_t - y_true

        self.d_V = np.dot(d_o_t, self.h_t.T)

        d_h_t = np.zeros((self.m, n))
        d_a_t = np.zeros((self.m, n))

        for t in range(n - 1, -1, -1):
            if t is n - 1:
                d_h_t[:, t] = np.dot(self.V.T, d_o_t[:, t])
            else:
                d_h_t[:, t] = np.dot(self.V.T, d_o_t[:, t]) + np.dot(self.W.T, d_a_t[:, t + 1])

            d_a_t[:, t] = np.dot(d_h_t[:, t].reshape(1, -1), np.diag(1 - np.tanh(np.power(self.a_t[:, t], 2))))

        self.d_W = np.dot(d_a_t[:, 1:], self.h_t[:, :-1].T)
        self.d_U = np.dot(d_a_t, self.X.T)

        self.d_b = d_a_t.sum(axis=1)
        self.d_c = d_o_t.sum(axis=1)

    def train(self, X_train, char_to_ind, n_epochs=7, seq_length=25, updates_per_log=10000, load_model=False):

        smooth_losses = []
        self.dW_sum = np.zeros(self.W.shape)
        self.dU_sum = np.zeros(self.U.shape)
        self.dV_sum = np.zeros(self.V.shape)
        self.db_sum = np.zeros(self.b.shape)
        self.dc_sum = np.zeros(self.c.shape)

        self.best_loss = 300

        if load_model is True:
            self.load_model()
        for i in range(n_epochs):
            hprev = np.zeros((self.m, 1))

            for e in range(0, len(X_train), seq_length):  #

                if e + seq_length > len(X_train):
                    print("Pame epomenh epoxh")
                    continue

                x_batch = to_one_hot(X_train[e:seq_length + e], char_to_ind)
                y_batch = to_one_hot(X_train[e + 1:seq_length + e + 1], char_to_ind)

                y_pred, _, loss = self.forward(x=x_batch, y_true=y_batch, h_t=hprev)
                hprev = self.h_t[:, -1].reshape(-1, 1)
                self.backward(y_true=y_batch)
                self.update_weights()

                if e == 0 and i == 0:
                    smooth_loss = loss
                else:
                    smooth_loss = .999 * smooth_loss + .001 * loss

                if smooth_loss < self.best_loss:
                    self.best_loss = smooth_loss
                    self.save_model()

                smooth_losses.append(smooth_loss)

                if (e / seq_length % updates_per_log == 0):
                    print(f"epoch = {i + 1}, iter = {e / seq_length}, Smooth loss = {smooth_loss}")
                    synthesized_text = self.synthesize(h_t=hprev, x=x_batch[:, 0], ind_to_char=ind_to_char)
                    print(synthesized_text)

        return smooth_losses

    def update_weights(self):
        self.dW_sum += np.power(self.d_W, 2)
        self.dU_sum += np.power(self.d_U, 2)
        self.dV_sum += np.power(self.d_V, 2)
        self.db_sum += np.power(self.d_b, 2).reshape(-1, 1)
        self.dc_sum += np.power(self.d_c, 2).reshape(-1, 1)

        self.W -= np.multiply(self.eta / np.sqrt(self.dW_sum + self.epsilon), self.d_W)
        self.U -= np.multiply(self.eta / np.sqrt(self.dU_sum + self.epsilon), self.d_U)
        self.V -= np.multiply(self.eta / np.sqrt(self.dV_sum + self.epsilon), self.d_V)
        self.b -= np.multiply(self.eta / np.sqrt(self.db_sum + self.epsilon), self.d_b.reshape(-1, 1))
        self.c -= np.multiply(self.eta / np.sqrt(self.dc_sum + self.epsilon), self.d_c.reshape(-1, 1))

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__ = state

    def load_model(self):
        self.__setstate__(load_model_params())

    def save_model(self):
        save_model_params(self.__getstate__())


if __name__ == "__main__":
    book, char_to_ind, ind_to_char = load_book()

    rnn = Vanila_RNN(m=100, K=len(char_to_ind))
    loss = rnn.train(X_train=book, char_to_ind=char_to_ind, n_epochs=20, seq_length=25, updates_per_log=10000)

    plot_loss(loss)

    print(f'BEST LOSS {rnn.best_loss}')
    h_0 = np.zeros((100, 1))
    x_0 = to_one_hot('.', char_to_ind)

    synthesized_text = rnn.synthesize(h_t=h_0, x=x_0, ind_to_char=ind_to_char, seq_length=1000)

    print(synthesized_text)
