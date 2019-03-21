import numpy as np
import Utils

class Classifier(object):
    def __init__(self, train_data, train_labels):
        self.train_data = train_data
        self.train_labels = train_labels
        self.weights, self.bias = self._initialize_weights(train_labels.shape[1], train_data.shape[1],
                                                           train_data.shape[0])

    def _initialize_weights(self, K, d, N):
        weigts = np.random.normal(0, 0.01, size=(K, d))
        bias = np.random.normal(0, 0.01, size=(K, N))

        return weigts, bias


if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test = Utils.load_cifar10()
    y_train = Utils.to_one_hot(y_train)
    y_val = Utils.to_one_hot(y_val)
    y_test = Utils.to_one_hot(y_test)

    model = Classifier(X_train, y_train)

    s = np.dot(model.weights, X_train.T) + model.bias
    s = Utils.softmax(s)
    pred = np.argmax(s,axis=0)

    print(pred)