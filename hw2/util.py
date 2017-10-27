from import_modules import *

class Preprocessor():
    def load_data(self, X_filename, Y_filename=None):
        data = {}
        X_df = pd.read_csv(X_filename, header=0)
        data["X"] = X_df.as_matrix()
        if Y_filename is not None:
            y_df = pd.read_csv(Y_filename, header=0)
            data["y"] = np.squeeze(y_df.as_matrix())
        return data

    def split(self, all_train):
        X = all_train['X']
        size = X.shape[0]
        indices = np.random.permutation(X.shape[0])
        # divid dataset to (train:valid) = (7:3) 
        point = (size * 7 ) // 10
        train = {}
        valid = {}
        train['X'] = np.take(all_train['X'], indices[:point], axis=0)
        train['y'] = np.take(all_train['y'], indices[:point], axis=0)
        valid['X'] = np.take(all_train['X'], indices[point:], axis=0)
        valid['y'] = np.take(all_train['y'], indices[point:], axis=0)
        return train, valid

    def __init__(self):
        pass

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
    
    for i in range(X.shape[0] // batch_size):
        X_batch = X_new[i * batch_size: (i+1) * batch_size]
        y_batch = None
        if y is not None:
            y_batch = y_new[i * batch_size: (i+1) * batch_size]
        yield {"X": X_batch, "y": y_batch}

def print_to_csv(y_, filename):
    d = {"id":[i + 1 for i in range(len(y_))],"label":y_}
    df = pd.DataFrame(data=d)
    df.to_csv(filename, index=False)

if __name__ == '__main__':
    p = Preprocessor()

    data = p.load_data("./data/X_train", "./data/Y_train")
    print(data['X'].shape)