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

    plt.figure()
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.figure()
    plt.plot(train_acc)
    plt.plot(val_acc)


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



def test_3_layer_networkd():
    np.seterr(all='raise')
    #
    X_train, y_train, X_val, y_val, X_test, y_test = Utils.load_cifar10(whole_dataset=True)
    X_train, mu, sigma = Utils.normalize_data(X_train)
    X_val, *_ = Utils.normalize_data(X_val, mu, sigma)
    X_test, *_ = Utils.normalize_data(X_test, mu, sigma)

    N = X_train.shape[0]
    d = X_train.shape[1]

    print(N, d)
    train_examples = N
    dim_size = d

    X_train = X_train[:train_examples, :dim_size]
    y_train = y_train[:train_examples, :dim_size]
    X_val = X_val[:train_examples, :dim_size]
    y_val = y_val[:train_examples, :dim_size]

    noc = 2
    l_min = -5
    l_max = -1
    cyclical_values = {'eta_min': 1e-5, 'eta_max': 1e-1, 'noc': noc, 'k': 5}

    param = {}
    # l = l_min + (l_max - l_min) * np.random.random_sample()
    # l = np.power(10, l)
    #
    # param['l'] = round(l, 6)
    l = 0.005
    start_time = time.time()

    dense1 = Dense(input_size=dim_size, output_size=50, l2_regul=l, std=1 / np.sqrt(dim_size))
    dense2 = Dense(input_size=50, output_size=50, l2_regul=l, std=1 / np.sqrt(50))
    dense3 = Dense(input_size=50, output_size=10, l2_regul=l, std=1 / np.sqrt(50))

    model = Classifier()
    model.add_layer(dense1)
    model.add_layer(ReLU())
    model.add_layer(dense2)
    model.add_layer(ReLU())
    model.add_layer(dense3)
    model.add_layer(SoftMax())

    train_loss, val_loss, train_acc, val_acc = model.fit(X_train, y_train,
                                                         X_val, y_val,
                                                         cyclical_values=cyclical_values,
                                                         batch_size=100)

    accuracy = model.predict(X_test, y_test)
    print("test accuracy", accuracy)
    end_time = time.time() - start_time

    # param['accuracy on test'] = accuracy
    param['accuracy on validation'] = np.amax(val_acc)
    print("validation maxed accuracy", np.amax(val_acc))

    param['number of cycles'] = noc
    param['time'] = round(end_time, 2)

    # Utils.write_to_csv(param)

    plt.figure()
    plt.title('Cost for lambda: {} cycles: {}'.format(l, noc))
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.legend(['train_loss,val_loss'])

    plt.figure()
    plt.title('Accuracy for lambda: {} cycles: {}'.format(l, noc))
    plt.plot(train_acc)
    plt.plot(val_acc)
    plt.legend(['train_acc,val_acc'])
    plt.show()


def test_deep_network():
    np.seterr(all='raise')
    #
    X_train, y_train, X_val, y_val, X_test, y_test = Utils.load_cifar10(whole_dataset=True)
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

    noc = 2
    l_min = -5
    l_max = -1
    cyclical_values = {'eta_min': 1e-5, 'eta_max': 1e-1, 'noc': noc, 'k': 5}

    param = {}
    # l = l_min + (l_max - l_min) * np.random.random_sample()
    # l = np.power(10, l)
    #
    # param['l'] = round(l, 6)
    l = 0.005
    start_time = time.time()

    dense1 = Dense(input_size=dim_size, output_size=50, l2_regul=l, std=1 / np.sqrt(dim_size))
    dense2 = Dense(input_size=50, output_size=20, l2_regul=l, std=1 / np.sqrt(30))
    dense3 = Dense(input_size=20, output_size=20, l2_regul=l, std=1 / np.sqrt(20))
    dense4 = Dense(input_size=20, output_size=20, l2_regul=l, std=1 / np.sqrt(20))
    dense5 = Dense(input_size=20, output_size=10, l2_regul=l, std=1 / np.sqrt(10))
    dense6 = Dense(input_size=10, output_size=10, l2_regul=l, std=1 / np.sqrt(10))
    dense7 = Dense(input_size=10, output_size=10, l2_regul=l, std=1 / np.sqrt(10))
    dense8 = Dense(input_size=10, output_size=10, l2_regul=l, std=1 / np.sqrt(10))
    dense9 = Dense(input_size=10, output_size=10, l2_regul=l, std=1 / np.sqrt(10))

    model = Classifier()
    model.add_layer(dense1)
    model.add_layer(ReLU())
    model.add_layer(dense2)
    model.add_layer(ReLU())
    model.add_layer(dense3)
    model.add_layer(ReLU())
    model.add_layer(dense4)
    model.add_layer(ReLU())
    model.add_layer(dense5)
    model.add_layer(ReLU())
    model.add_layer(dense6)
    model.add_layer(ReLU())
    model.add_layer(dense7)
    model.add_layer(ReLU())
    model.add_layer(dense8)
    model.add_layer(ReLU())
    model.add_layer(dense9)
    model.add_layer(SoftMax())

    train_loss, val_loss, train_acc, val_acc = model.fit(X_train, y_train,
                                                         X_val, y_val,
                                                         cyclical_values=cyclical_values,
                                                         batch_size=100)

    accuracy = model.predict(X_test, y_test)
    print("test accuracy", accuracy)
    end_time = time.time() - start_time

    # param['accuracy on test'] = accuracy
    param['accuracy on validation'] = np.amax(val_acc)
    print("validation maxed accuracy", np.amax(val_acc))

    param['number of cycles'] = noc
    param['time'] = round(end_time, 2)

    Utils.write_to_csv(param)

    plt.figure()
    plt.title('Cost for lambda: {} cycles: {}'.format(l, noc))
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.legend(['train_loss,val_loss'])

    plt.figure()
    plt.title('Accuracy for lambda: {} cycles: {}'.format(l, noc))
    plt.plot(train_acc)
    plt.plot(val_acc)
    plt.legend(['train_acc,val_acc'])
    plt.show()
