from import_modules import *
from util import *
from model import *
class Config():
    def __init__(self, args):
        # feature matrix
        self.batch_size = args.batch_size
        self.n_epochs = args.n_epochs
        self.lr = args.lr
        self.option = args.option
        self.param_folder = args.param_folder
        self.load_model = args.load_model

def main():
    # parse argument
    parser = argparse.ArgumentParser(description='ML2017/hw1')
    parser.add_argument('option', type=str, help='option')
    parser.add_argument('--train_filename', type=str, help='training_label.txt', default=None)
    parser.add_argument('--train_nolabel_filename', type=str, help='training_nolabel.txt', default=None)
    parser.add_argument('--test_filename', type=str, help='test.csv', default=None)
    parser.add_argument('--testout', type=str, help="test_out.csv", default=None)
    parser.add_argument('--param_folder', type=str, default="./param")
    parser.add_argument('--htc', type=bool, default=False)
    parser.add_argument('--n_epochs', type=int, default=1000,
                        help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch size, -1 denote train on all training data')
    parser.add_argument('--load_model', type=bool, default=False)
    args = parser.parse_args()
    
    # init model
    now = dt.datetime.now()
    timeline = "time: %d%02d%02d_%02d%02d" % (now.year, now.month, now.day, now.hour, now.minute)
    print("=" * 80 + "\nModel init\n" + "=" * 80)
    config = Config(args)
    model = SentimentClassifier(config)

    
    # if Open ai platform
    if args.htc == True:
        args.train_filename = os.path.join(os.environ.get("GRAPE_DATASET_DIR"), args.train_filename)
        args.test_filename = os.path.join(os.environ.get("GRAPE_DATASET_DIR"), args.test_filename)
        args.testout = os.path.join(args.param_folder, args.testout)
    print("=" * 80 + "\nPreprocess\n" + "=" * 80)
    
    # preprocess
    p = Preprocess()
    train_df, test_df, train_df_nolabel = \
    p.read_txt(args.train_filename, args.test_filename, args.train_nolabel_filename)

    # load old train_df and valid df
    train_df, valid_df = model.load_presplit()

    '''
    TODO: Semi-supervised
    1. fit
    2. predict train_df_nolabel
    3. put train_df_nolabel whose label confidence bigger than 80% into train_df
    4. got to step 1
    '''
    # fit
    print("=" * 80 + "\nModel fit\n" + "=" * 80)
    if False:
        for _ in range(3):
            train_accuracy_list, valid_accuracy_list, max_valid_accuracy, train_losses, valid_losses = \
            model.fit(train_df, valid_df, train_df_nolabel, timeline)
            y_, argmax_y_ = model.predict(train_df_nolabel)
            pdb.set_trace()
            # retrieve row 
            retrieve_array = np.logical_or(y_[:,0] > 0.9, y_[:,1] > 0.9)
            print("Number of new label {}".format(np.sum(retrieve_array)))
            # indices of True 
            retrieve_array = retrieve_array.nonzero()[0]
            # retrieve df
            retrieve_df = pd.DataFrame({"label":argmax_y_[retrieve_array], "seq":train_df_nolabel["seq"].iloc[retrieve_array]})
            # put new training data
            train_df = pd.concat([train_df, retrieve_df])
            # drop label
            train_df_nolabel.drop([train_df_nolabel.index[index] for index in retrieve_array], inplace=True)
    else:

        train_accuracy_list, valid_accuracy_list, max_valid_accuracy, train_losses, valid_losses = \
            model.fit(train_df, valid_df, train_df_nolabel, timeline)
    # predict test data
    if test_df is not None:
        test = {}
        test['y_'], test['argmax_y_'] = model.predict(test_df)
        
        print_to_csv(test['argmax_y_'], args.testout)

    #
    '''
    experiment = {}
    experiment_df = pd.DataFrame({"id":[0,1], "seq":["today is a day, but it is hot", \
        "today is hot, but it is a day"]})
    embed()
    experiment['y_'], experiment['argmax_y_'] = model.predict(experiment_df)
    print_to_csv(experiment['y_'], "experiment")
    '''
    # Dump training pickle file
    '''
    with open("./train_accuracy_list", "wb") as f, open("./valid_accuracy_list", "wb") as f2, \
    open("./train_losses", "wb") as f3, open("./valid_losses", "wb") as f4:
        pickle.dump(train_accuracy_list, f)
        pickle.dump(valid_accuracy_list, f2)
        pickle.dump(train_losses, f3)
        pickle.dump(valid_losses, f4)
        
    with open("logfile", "a") as file:
        
        print(timeline, file=file)
        print("epoch:{}".format(args.n_epochs), file=file)
        print("lr:{}".format(args.lr), file=file)
        print("batch_size:{}".format(args.batch_size), file=file)
        print("option:{}".format(args.option), file=file)
        print("valid accuracy: {}".format(max_valid_accuracy), file=file)
        print("", file=file)
    '''
    return 0

if __name__ == '__main__':
    main()