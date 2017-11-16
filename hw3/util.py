from import_modules import *
class Preprocess():
    
    def read_csv(self, train_filename, test_filename):
        train_df = None
        test_df = None
        if train_filename is not None:
            train_df = pd.read_csv(train_filename, header=0)
        if test_filename is not None:
            test_df = pd.read_csv(test_filename, header=0)
        return train_df, test_df

    def create_numpy(self, train_df, test_df):
        # init numpy dict
        train = None
        test = None

        # split element in feature column
        if train_df is not None:
            train = {}
            train_X_series = train_df['feature'].apply(lambda x: x.split())
            train_y_series = train_df['label']
            train['X'] = np.reshape(np.stack(train_X_series, axis=0).astype(float), [-1, 48, 48])
            train['y'] = np.stack(train_y_series, axis=0)

        # test
        if test_df is not None:
            test = {}
            test_X_series = test_df['feature'].apply(lambda x: x.split())
            test['X'] = np.reshape(np.stack(test_X_series, axis=0).astype(float), [-1, 48, 48])
        return train, test

    def load_data(self):
        # read_csv
        train_df, test_df = self.read_csv(self.args.train_filename, self.args.test_filename)
        
        # create numpy
        train, test = self.create_numpy(train_df, test_df)
        
        return train, test 

    def __init__(self, args):
        self.args = args

def batch_generator(X, y, batch_size, shuffle):
    if batch_size == -1:
        yield {"X": X, "y": y}
        return
    if shuffle:
        # generate permutation
        indices = np.random.permutation(X.shape[0])
        # Just replace it
        X_new = np.take(X, indices, axis=0)
        y_new = np.take(y, indices, axis=0)
    else:
        X_new = X
        y_new = y
    quotient = X.shape[0] // batch_size
    remainder = X.shape[0] - quotient * batch_size
    for i in range(quotient):
        X_batch = X_new[i * batch_size: (i+1) * batch_size]
        y_batch = None
        if y is not None:
            y_batch = y_new[i * batch_size: (i+1) * batch_size]
        yield {"X": X_batch, "y": y_batch}

    if remainder > 0:
        X_batch = X_new[quotient * batch_size:]
        y_batch = None
        if y is not None:
            y_batch = y_new[quotient * batch_size:]
        yield {"X": X_batch, "y": y_batch}


def print_to_csv(y_, filename):
    d = {"id":[i for i in range(len(y_))],"label":y_}
    df = pd.DataFrame(data=d)
    df.to_csv(filename, index=False)