import numpy as np
import pandas as pd
import random
import argparse
import datetime as dt
from util import Loader, print_to_csv, plot_func
from model import LinearRegressionModel
from dnn import DNN
import matplotlib.pyplot as plt
class Config():
    def __init__(self, args, loader):
        # feature matrix
        self.feature = 18 * 9
        self.batch_size = args.batch_size
        self.n_epochs = args.n_epochs
        self.lr = args.lr
        self.loader = loader
        self.option = args.option
        self.regularize = args.regularize
        self.load_prev = args.load_prev
        self.timestamp = args.timestamp
        self.normalize = args.normalize

def main(args, debug=True, write_to_file=True, write_to_log=True):
    # initial value
    #random.seed(314)
    #np.random.seed(567)
    now = dt.datetime.now()
    
    # load training data
    loader = Loader(args.train, args.label, args.batch_size)
    
    # set up training data and validation data
    train = {}
    valid = {}
    all_train = {}
    test = {}
    train['X'], train['y'] = loader.load_numpy_data(args.train, args.label)
    test['X'] = loader.load_test_data(args.testcsv)
    valid['X'], valid['y'] = loader.load_numpy_data(args.valid_train, args.valid_label)
    all_train['X'] = np.concatenate([train['X'], valid['X']], axis=0)
    all_train['y'] = np.concatenate([train['y'], valid['y']], axis=0)
    
    # config object
    config = Config(args, loader)
    
    # set up model
    train_loss = valid_loss = all_train_loss = None
    model = LinearRegressionModel(config)
    if args.load_prev == False: 
        train_loss, valid_loss = model.fit(train, valid)
        # fit all data
        all_train_loss, _ = model.fit(all_train, None)
    
    # save param
    param = ['w', 'b', 'mu', 'sd']
    filename = ["./param/%s_%d%02d%02d_%02d%02d" % (p, now.year, now.month, now.day, now.hour, now.minute) for p in param]
    if args.load_prev == False:
        model.save_param(filename)
    
    # write log file
    if write_to_log and args.load_prev == False:
        with open("adam_record_train_valid", "a") as file:
            timeline = "time: %d%02d%02d_%02d%02d" % (now.year, now.month, now.day, now.hour, now.minute)
            print(timeline, file=file)
            print("epoch:{}".format(args.n_epochs), file=file)
            print("lr:{}".format(args.lr), file=file)
            print("batch_size:{}".format(args.batch_size), file=file)
            print("option:{}".format(args.option), file=file)
            print("regularization:{}".format(args.regularize), file=file)
            print("train loss: {}".format(train_loss), file=file)
            print("validation loss: {}".format(valid_loss), file=file)
            print("total train loss: {}".format(all_train_loss), file=file)
            print("train data: {}".format(args.train), file=file)
            print("valid data: {}".format(args.valid_train), file=file)
            print("", file=file)
    # write test output file
    if write_to_file:
        # predict test
        test['y_'] = model.predict(test)
        # print to csv file
        print_to_csv(test['y_'], args.testout)
    # check closed form
    # A w = b
    if debug == True:
        tmp = np.reshape(train['X'], (train['X'].shape[0], -1))
        tmp = np.concatenate([tmp, np.ones((tmp.shape[0], 1))], axis=1)
        print(tmp.shape)
        x_, residuals, rank, s = np.linalg.lstsq(tmp, train['y'])
        print(np.sqrt(residuals / train['X'].shape[0]))
    return train_loss, valid_loss, all_train_loss

if __name__ == '__main__':
    # parse argument
    parser = argparse.ArgumentParser(description='ML2017/hw1')
    parser.add_argument('option', type=str, help='option')
    parser.add_argument('regularize', type=float, help='regularize')
    parser.add_argument('train', type=str, help='train_numpy')
    parser.add_argument('label', type=str, help='label_numpy')
    parser.add_argument('valid_train', type=str, help='valid_train')
    parser.add_argument('valid_label', type=str, help='valid_label')
    parser.add_argument('testcsv', type=str, help='test.csv')
    parser.add_argument('testout', type=str, help="test_out.csv")
    #parser.add_argument('model', type=str, help='Linear Regression')
    parser.add_argument('--n_epochs', type=int, default=1000,
                        help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.5,
                        help='learning rate')
    parser.add_argument('--batch_size', type=int, default=180,
                        help='batch size')
    parser.add_argument('--load_prev', type=bool, default=False,
                        help='load_prev')
    parser.add_argument('--timestamp', type=str, help='timestamp ex.20171002_1743')
    parser.add_argument('--name', type=str, default='NN',
                        help='name for the model, will be the directory name for summary')
    parser.add_argument('--normalize', type=str, default="Z",
                        help='normalize type')
    args = parser.parse_args()
    main(args, debug=False, write_to_file=True)
    if False:
        lines = ["PM2.5_power2_NO {} train_numpy label_numpy ./test.csv ./testout".format(0.1 ** i) for i in range(0, 4+1)] 
        regu = [0.1 ** i for i in range(0, 4+1)]
        train_losses = []
        valid_losses = []
        for line in lines:
            args = parser.parse_args(line.split())
            _, valid_loss, all_train_loss = main(args, False, False)
            train_losses.append(all_train_loss)
            valid_losses.append(valid_loss)
        plot_func(regu, train_losses, valid_losses)