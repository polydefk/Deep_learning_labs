from model import *
import Utils


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

    plt.figure()
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.figure()
    plt.plot(train_acc)
    plt.plot(val_acc)


def test_overfiting_without_regul():
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

    Utils.plot_loss_acc(train_loss, train_acc, val_loss, val_acc)



def test_grad_checking_both_layers():
    X_train, y_train, X_val, y_val, X_test, y_test = Utils.load_cifar10(whole_dataset=False)
    batch_size = 20
    dim_size = 500
    X_train, mu, sigma = Utils.normalize_data(X_train)
    X_val, *_ = Utils.normalize_data(X_val, mu, sigma)
    X_test, *_ = Utils.normalize_data(X_test, mu, sigma)

    X_train = X_train[:batch_size, :dim_size]
    y_train = y_train[:batch_size]

    dense1 = Dense(input_size=dim_size, output_size=15, l2_regul=0., std=1 / np.sqrt(dim_size))
    dense2 = Dense(input_size=15, output_size=10, l2_regul=0, std=1 / np.sqrt(50))

    model = Classifier()
    model.add_layer(dense1)
    model.add_layer(ReLU())
    model.add_layer(dense2)
    model.add_layer(SoftMax())

    model.forward_pass(X_train)
    model.backward_pass(y_train)

    weights = [dense1.W, dense2.W]
    grad_w = [dense1.grad_w, dense2.grad_w]

    Utils.compute_grads_for_matrix(y_train, X_train, weights, model, grad_w)



def test_Batch_normalization():
    X_train, y_train, X_val, y_val, X_test, y_test = Utils.load_cifar10(whole_dataset=False)
    batch_size = 20
    dim_size = 500
    X_train, mu, sigma = Utils.normalize_data(X_train)
    X_val, *_ = Utils.normalize_data(X_val, mu, sigma)
    X_test, *_ = Utils.normalize_data(X_test, mu, sigma)

    X_train = X_train[:batch_size, :dim_size]
    y_train = y_train[:batch_size]

    dense1 = Dense(input_size=dim_size, output_size=15, l2_regul=0., std=1 / np.sqrt(dim_size))
    dense2 = Dense(input_size=15, output_size=10, l2_regul=0, std=1 / np.sqrt(50))

    model = Classifier()
    model.add_layer(dense1)
    model.add_layer(ReLU())
    model.add_layer(dense2)
    model.add_layer(SoftMax())

    model.forward_pass(X_train)