from import_modules import *
from util import *
from model import *
class Config():
    def __init__(self, kmeans_path, param_folder):
        self.method = "autoencoder"
        self.kmeans_path = kmeans_path
        self.param_folder = param_folder

def main():
    parser = argparse.ArgumentParser(description='ML2017/hw6')
    parser.add_argument('image_npy_path', type=str, help='image.npy path')
    parser.add_argument('test_case_path', type=str, help="test_case.csv path")
    parser.add_argument('testout', type=str, help="prediction file path")
    parser.add_argument('kmeans_path', type=str, help="kmeans.pkl")
    parser.add_argument('param_folder', type=str, help="param_folder")
    args = parser.parse_args()
    # timeline 
    now = dt.datetime.now()
    timeline = "time: %d%02d%02d_%02d%02d" % (now.year, now.month, now.day, now.hour, now.minute)
    # 
    image_array = load_image_npy(args.image_npy_path)
    test_case_df = load_test_case_csv(args.test_case_path)
    # normalize image
    mean = np.mean(image_array, axis=0)
    std = np.std(image_array, axis=0, ddof=1)
    #pdb.set_trace()
    image_array = (image_array - mean) / (std + 1e-6)
    # 
    config = Config(args.kmeans_path, args.param_folder)
    model = Is_it_same_classifier(config)
    model.fit(image_array)
    ans = model.predict(test_case_df)
    # print to csv
    print_to_csv(ans, args.testout)
if __name__ == '__main__':
    main()