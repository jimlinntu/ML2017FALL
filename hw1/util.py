import numpy as np
class Loader(object):
    def load_train_data(self, replace_NR=True):
        X = np.load(self.train_name)
        y = np.load(self.label_name)
        if replace_NR:
            X[X == 'NR'] = 0
        X = X.astype(np.float)
        y = y.astype(np.float)
        return X, y
    def split(self, X, y):
        size = X.shape[0]
        indices = np.random.permutation(X.shape[0])
        # divid dataset to (train:valid) = (7:3) 
        point = (size * 7 )// 10
        X_train = np.take(X, indices[:point], axis=0)
        y_train = np.take(y, indices[:point], axis=0)
        X_valid = np.take(X, indices[point+1:], axis=0)
        y_valid = np.take(y, indices[point+1:], axis=0)
        return X_train, y_train, X_valid, y_valid

    def batch_generator(self, X, y):
        if self.shuffle:
            # generate permutation
            indices = np.random.permutation(X.shape[0])
            # Just replace it
            X = np.take(X, indices, axis=0)
            y = np.take(y, indices, axis=0)
        assert X.shape[0] % self.batch_size == 0
        batch_size = self.batch_size
        for i in range(X.shape[0] // batch_size):
            X_batch = X[i * batch_size: (i+1) * batch_size]
            y_batch = None
            if y is not None:
                y_batch = y[i * batch_size: (i+1) * batch_size]
            yield {"X": X_batch, "y": y_batch}

    def __init__(self, train_name, label_name, batch_size, shuffle=True):
        self.train_name = train_name
        self.label_name = label_name
        self.batch_size = batch_size
        self.shuffle = shuffle