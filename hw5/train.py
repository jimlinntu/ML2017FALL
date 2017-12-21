from import_modules import *
from util import *
from model import *
class Config():
    def __init__(self, args):
        # feature matrix
        self.batch_size = args.batch_size
        self.n_epochs = args.n_epochs
        self.lr = args.lr
        self.param_folder = args.param_folder
        self.load_model = args.load_model
        self.latent_dim = args.latent_dim
        self.bias = args.bias
def main():
    parser = argparse.ArgumentParser(description='ML2017/hw5')
    parser.add_argument('--train', type=str, help='train.csv', default=None)
    parser.add_argument('--test', type=str, help='test.csv', default=None)
    parser.add_argument('--testout', type=str, help='testout', default=None)
    parser.add_argument('--load_model', type=bool, default=False)
    parser.add_argument('--param_folder', type=str, default="./param")
    parser.add_argument('--n_epochs', type=int, default=1000,
                        help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='batch size, -1 denote train on all training data')
    parser.add_argument("--latent_dim", type=int, default=60)
    parser.add_argument("--bias", type=bool, default=False)
    args = parser.parse_args()
    # timeline 

    now = dt.datetime.now()
    timeline = "time: %d%02d%02d_%02d%02d" % (now.year, now.month, now.day, now.hour, now.minute)
    # config
    config = Config(args)
    # read dataset
    if args.train is not None:
        train = read_train(args.train, args.test)
        userid2index, movies2index = train["userid2index"], train["movies2index"]
        pickle.dump(userid2index, open("./param/userid2index", "wb"))
        pickle.dump(movies2index, open("./param/movies2index", "wb"))
    else:
        # load userid2index movies2index
        userid2index = pickle.load(open("./param/userid2index", "rb"))
        movies2index = pickle.load(open("./param/movies2index", "rb"))
    test = read_test(args.test, userid2index, movies2index)
    # model
    model = RatingClassifier(config)
    if args.train is not None:
        # model split train valid
        train, valid = model.split(train)
    else:
        train, valid = None, None
    # fit
    training_accuracy_list, valid_accuracy_list, min_valid_loss, train_loss_list, \
        valid_loss_list = model.fit(train, valid, timeline)

    
    # predict
    test["y_"] = model.predict(test["X"])
    # print to csv
    print_to_csv(test["y_"], args.testout)
    # write log
    with open("logfile", "a") as file:
        print(timeline, file=file)
        print("epoch:{}".format(args.n_epochs), file=file)
        print("lr:{}".format(args.lr), file=file)
        print("batch_size:{}".format(args.batch_size), file=file)
        print("min_valid_loss: {}".format(min_valid_loss), file=file)
        print("", file=file)
if __name__ == '__main__':
    import sys, traceback, pdb
    try:
        main()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)