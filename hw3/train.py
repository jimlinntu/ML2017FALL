from import_modules import *
from util import *
from model import *
class Config():
    def __init__(self, args):
        # feature matrix
        self.data_aug = args.data_aug
        self.batch_size = args.batch_size
        self.n_epochs = args.n_epochs
        self.lr = args.lr
        self.option = args.option
        self.timestamp = args.timestamp
        self.cuda = False
        self.param_folder = args.param_folder
        self.log = args.log

def main():
    # parse argument
    parser = argparse.ArgumentParser(description='ML2017/hw1')
    parser.add_argument('option', type=str, help='option')
    parser.add_argument('--train_filename', type=str, help='train.csv', default=None)
    parser.add_argument('--test_filename', type=str, help='test.csv', default=None)
    parser.add_argument('--testout', type=str, help="test_out.csv", default=None)
    parser.add_argument('--log', type=str, default="./log")
    parser.add_argument('--data_aug', type=bool, default=False)
    parser.add_argument('--param_folder', type=str, default="./param")
    parser.add_argument('--htc', type=bool, default=False)
    parser.add_argument('--n_epochs', type=int, default=1000,
                        help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size, -1 denote train on all training data')
    parser.add_argument('--timestamp', type=str, help='timestamp ex.20171002_1743')
    args = parser.parse_args()
    
    # init model
    now = dt.datetime.now()
    timeline = "time: %d%02d%02d_%02d%02d" % (now.year, now.month, now.day, now.hour, now.minute)
    print("=" * 80 + "\nModel init\n" + "=" * 80)
    config = Config(args)
    model = BaseLineCNNWrapper(config)

    
    # if Open ai platform
    if args.htc == True:
        args.train_filename = os.path.join(os.environ.get("GRAPE_DATASET_DIR"), args.train_filename)
        args.test_filename = os.path.join(os.environ.get("GRAPE_DATASET_DIR"), args.test_filename)
        args.testout = os.path.join(args.param_folder, args.testout)
    print("=" * 80 + "\nPreprocess\n" + "=" * 80)
    
    # preprocess
    p = Preprocess(args)
    train, test = p.load_data()

    # fit
    print("=" * 80 + "\nModel fit\n" + "=" * 80)
    if train is not None:
        train_accuracy, valid_accuracy = model.fit(train, None, timeline)

    # predict test data
    if test is not None:
        # load model
        model.model.load_state_dict(torch.load(os.path.join(args.param_folder, args.timestamp)))
        # load normalize value
        with open(os.path.join(args.param_folder, args.timestamp) + "mean", "rb") as f:
            model.mean = pickle.load(f)

        with open(os.path.join(args.param_folder, args.timestamp) + "std", "rb") as f:
            model.std = pickle.load(f)

        test['y_'] = model.predict(test['X'])

        print_to_csv(test['y_'], args.testout)
    
    '''
    if args.load_prev == False:
        
        with open("logfile", "a") as file:
            
            print(timeline, file=file)
            print("epoch:{}".format(args.n_epochs), file=file)
            print("lr:{}".format(args.lr), file=file)
            print("batch_size:{}".format(args.batch_size), file=file)
            print("option:{}".format(args.option), file=file)
            print("regularization:{}".format(args.regularize), file=file)
            print("train accuracy: {}".format(train_accuracy), file=file)
            print("valid accuracy: {}".format(valid_accuracy), file=file)
            #print("total train loss: {}".format(all_train_loss), file=file)
            print("", file=file)
    '''
    return 0

if __name__ == '__main__':
    main()