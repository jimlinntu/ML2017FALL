import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
class Loader(object):
    def load_numpy_data(self, train_name, label_name, replace_NR=True,):
        X = np.load(train_name)
        y = np.load(label_name)
        if replace_NR:
            X[X == 'NR'] = 0
        X = X.astype(np.float)
        y = y.astype(np.float)
        return X, y
    def split(self, X, y):
        size = X.shape[0]
        indices = np.random.permutation(X.shape[0])
        # divid dataset to (train:valid) = (7:3) 
        point = (size * 7 ) // 10
        X_train = np.take(X, indices[:point], axis=0)
        y_train = np.take(y, indices[:point], axis=0)
        X_valid = np.take(X, indices[point:], axis=0)
        y_valid = np.take(y, indices[point:], axis=0)


        return X_train, y_train, X_valid, y_valid

    def batch_generator(self, X, y):
        if self.shuffle:
            # generate permutation
            indices = np.random.permutation(X.shape[0])
            # Just replace it
            X = np.take(X, indices, axis=0)
            y = np.take(y, indices, axis=0)
        #assert X.shape[0] % self.batch_size == 0
        batch_size = self.batch_size
        for i in range(X.shape[0] // batch_size):
            X_batch = X[i * batch_size: (i+1) * batch_size]
            y_batch = None
            if y is not None:
                y_batch = y[i * batch_size: (i+1) * batch_size]
            yield {"X": X_batch, "y": y_batch}

    def load_test_data(self, filename, replace_NR=True):
        metric = 18
        test_df = pd.read_csv(filename, header=None, encoding="utf-8")
        metric_names = [test_df[1][i] for i in range(metric)]
        X = []
        for i in range(test_df.shape[0] // metric):
            X.append(test_df.iloc[i * metric: (i+1) * metric, 2:].values)
        X = np.array(X)
        if replace_NR:
            X[X == 'NR'] = 0
        
        #print(X)
        X = X.astype(np.float)
        return X
    def __init__(self, train_name, label_name, batch_size, shuffle=True):
        self.train_name = train_name
        self.label_name = label_name
        self.batch_size = batch_size
        self.shuffle = shuffle

def print_to_csv(y_, filename):
    size = y_.shape[0]
    d = {'id': ["id_{}".format(i) for i in range(size)], 'value': y_}
    df = pd.DataFrame(data=d)
    df.to_csv(filename, index=False)

def plot_func(regu, train_losses, valid_losses):
    plt.xlabel("lambda")
    plt.ylabel("RMSE(without regularization loss)")
    plt.xscale('log')
    plt.title("Regularization lambda with true RMSE(wihtout regularization loss)")
    plt.plot(regu, train_losses, "r",label="Train")
    plt.plot(regu, valid_losses, "g", label="Valid")
    plt.legend(loc="upper center")
    plt.show("Regu")