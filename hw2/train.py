from import_modules import *
from util import Preprocessor, print_to_csv
from model import LogisticRegression, GenerativeModel, RandomForestModel

class Config():
    def __init__(self, args):
        # feature matrix
        self.batch_size = args.batch_size
        self.n_epochs = args.n_epochs
        self.lr = args.lr
        self.option = args.option
        self.regularize = args.regularize
        self.load_prev = args.load_prev
        self.timestamp = args.timestamp
        self.normalize = args.normalize

def main():
    # parse argument
    parser = argparse.ArgumentParser(description='ML2017/hw1')
    parser.add_argument('model', type=str, help='L(ogistic) or G(enerative) or T(orch) or RF(Random forest)')
    parser.add_argument('option', type=str, help='option')
    parser.add_argument('regularize', type=float, help='regularize')
    parser.add_argument('X_train', type=str, help='X_train')
    parser.add_argument('Y_train', type=str, help='Y_train')
    parser.add_argument('X_test', type=str, help='Y_train')
    parser.add_argument('testout', type=str, help="test_out.csv")
    parser.add_argument('--n_epochs', type=int, default=5000,
                        help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.02,
                        help='learning rate')
    parser.add_argument('--batch_size', type=int, default=-1,
                        help='batch size, -1 denote train on all training data')
    parser.add_argument('--load_prev', type=bool, default=False,
                        help='load_prev')
    parser.add_argument('--timestamp', type=str, help='timestamp ex.20171002_1743')
    parser.add_argument('--name', type=str, default='NN',
                        help='name for the model, will be the directory name for summary')
    parser.add_argument('--normalize', type=str, default="Z",
                        help='normalize type')
    args = parser.parse_args()
    
    # Init
    config = Config(args)
    preprocessor = Preprocessor()
    all_train = preprocessor.load_data(args.X_train, args.Y_train)
    train, valid = preprocessor.split(all_train)
    test = preprocessor.load_data(args.X_test, None)

    # Establish model
    if args.model == "L":
        model = LogisticRegression(config)
    elif args.model == "G":
        model = GenerativeModel(config)
    elif args.model == "T":
        model = Torch_Logistic_Regression(config)
    elif args.model == "RF":
        model = RandomForestModel(config)

    # If not load prev
    if args.load_prev == False: 
        train_loss, valid_loss = model.fit(all_train, None)
    else:
        model.load_param()

    
    test['y_'] = model.predict(test['X'])
    

    # save param
    now = dt.datetime.now()
    param = ['w', 'b', 'mu', 'sigma']
    filename = ["./param/%s_%d%02d%02d_%02d%02d" % 
    (p, now.year, now.month, now.day, now.hour, now.minute) for p in param]
    if args.load_prev == False and args.model != "G":
        model.save_param(filename)

    # write log file
    if args.load_prev == False:
        with open("adagrad_record", "a") as file:
            timeline = "time: %d%02d%02d_%02d%02d" % (now.year, now.month, now.day, now.hour, now.minute)
            print(timeline, file=file)
            print("epoch:{}".format(args.n_epochs), file=file)
            print("lr:{}".format(args.lr), file=file)
            print("batch_size:{}".format(args.batch_size), file=file)
            print("option:{}".format(args.option), file=file)
            print("regularization:{}".format(args.regularize), file=file)
            print("train loss: {}".format(train_loss), file=file)
            
            #print("total train loss: {}".format(all_train_loss), file=file)
            print("", file=file)
    
    # print to csv file
    print_to_csv(test['y_'], args.testout)

if __name__ == '__main__':
    main()