import numpy as np
import random
import argparse
from util import Loader
from model import LinearRegressionModel
class Config():
    def __init__(self, batch_size, n_epochs, lr, loader):
        # feature matrix
        self.feature = 18 * 9
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.lr = lr
        self.loader = loader

def main(debug=True):
    random.seed(314)
    np.random.seed(567)
    # parse argument
    parser = argparse.ArgumentParser(description='ML2017/hw1')
    parser.add_argument('train', type=str, help='train_numpy')
    parser.add_argument('label', type=str, help='label_numpy')
    #parser.add_argument('model', type=str, help='Linear Regression')
    parser.add_argument('--n_epochs', type=int, default=50,
                        help='number of epochs')
    parser.add_argument('--lr', type=float, default=1e-1,
                        help='learning rate')
    parser.add_argument('--batch_size', type=int, default=180,
                        help='batch size')
    parser.add_argument('--name', type=str, default='NN',
                        help='name for the model, will be the directory name for summary')
    args = parser.parse_args()
    # load training data
    loader = Loader(args.train, args.label, args.batch_size)
    # set up training data and validation data
    train = {}
    valid = {}
    train['X'], train['y'] = loader.load_train_data()
    train['X'], train['y'], valid['X'], valid['y'] = loader.split(train['X'], train['y'])
    config = Config(args.batch_size, args.n_epochs, args.lr, loader)
    # set up model
    model = LinearRegressionModel(config)
    valid_loss = model.fit(train, valid)
    print("Last validation loss is {}".format(valid_loss))
    # check closed form
    if debug == True:
        x_, residuals, rank, s= np.linalg.lstsq(np.reshape(train['X'], (train['X'].shape[0], -1)), train['y'])
        print(residuals / train['X'].shape[0])
if __name__ == '__main__':
    main()