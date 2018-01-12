from import_modules import *
from trainer import batch_generator
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
class Is_it_same_classifier():
    def __init__(self, config):
        self.config = config
    def fit(self, image_array):
        print("Model fitting..." + "." * 80)
        try:
            self.kmeans = pickle.load(open(self.config.kmeans_path, "rb"))
        except FileNotFoundError:
            reduced_image_array = self.dimension_reduction(image_array, self.config.method, dimension=32)
            self.kmeans = KMeans(n_clusters=2).fit(reduced_image_array)
            pickle.dump(self.kmeans, open(self.config.kmeans_path, "wb"))
        #color = ["blue", "red"]
        #color_list = [color[label] for label in self.kmeans.labels_]
        #plt.scatter(reduced_image_array[:, 0], reduced_image_array[:, 1], c=color_list)
        #plt.savefig("tsne")
    
    def predict(self, test_case_df):
        ans = [None] * test_case_df.shape[0]
        index2label = {index:label for index, label in enumerate(self.kmeans.labels_)}
        for index, row in enumerate(test_case_df.itertuples()):
            if index2label[row[2]] == index2label[row[3]]:
                ans[index] = 1  
            else:
                ans[index] = 0
        return ans


    def dimension_reduction(self, image_array, method, dimension=2):
        print("Dimension reducing..." + "." * 80)
        if method == "autoencoder":
            model = AutoEncoder_DNN2().cuda()
            model.load_state_dict(torch.load(os.path.join(self.config.param_folder, "besttime: 20180102_1519")))
            g = batch_generator(image_array, batch_size=300, shuffle=False, volatile=True)
            reduced_image_array = []
            pdb.set_trace()
            for batch in g:
                X_, code = model(batch["X"])
                reduced_image_array.append(code.cpu().data.numpy())
            reduced_image_array = np.concatenate(reduced_image_array, axis=0)
            print(reduced_image_array.shape)

        return reduced_image_array

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        # encode
        self.conv_1 = nn.Conv2d(1, 5, kernel_size=3, stride=1, padding=0)
        self.maxpool_1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        #self.dropout = nn.Dropout(p=0.3)
        #self.relu = nn.ReLU(inplace=True)
        self.conv_2 = nn.Conv2d(5, 10, kernel_size=2, stride=1, padding=0)
        self.maxpool_2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        # dense code
        self.dense = nn.Linear(360, 10)
        self.inverse_dense = nn.Linear(10, 360)
        # decode
        self.deconv_1 = nn.Conv2d(10, 5, kernel_size=2, stride=1, padding=1)
        self.unpool_1 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.deconv_2 = nn.Conv2d(5, 1, kernel_size=3, stride=1, padding=2)
        self.unpool_2 = nn.MaxUnpool2d(kernel_size=2, stride=2)

            

    def forward(self, X):
        X = X.view(-1, 1, 28, 28)
        batch_size = X.size()[0]
        # encode
        
        out = self.conv_1(X)
        out, indices_1 = self.maxpool_1(out)
        #out = self.dropout(out)
        #out = self.relu(out)
        out = self.conv_2(out)
        out, indices_2 = self.maxpool_2(out)

        code = self.dense(out.view(batch_size, -1))
        out = self.inverse_dense(code)
        # decode
        out = out.view(-1, 10, 6, 6)
        #out = self.dropout(out)
        out = self.unpool_1(out, indices_2)
        out = self.deconv_1(out)
        #out = self.dropout(out)
        out = self.unpool_2(out, indices_1)
        out = self.deconv_2(out)
        out = out.view(-1, 784)
        return out, code

class AutoEncoder_DNN(nn.Module):
    def __init__(self):
        super(AutoEncoder_DNN, self).__init__()
        self.encode = nn.Sequential(
                nn.Linear(784, 500),
                nn.Linear(500, 300),
                nn.Linear(300, 200),
                nn.Linear(200, 100),
                nn.Linear(100, 32),
            )
        self.decode = nn.Sequential(
                nn.Linear(32, 100),
                nn.Linear(100, 200),
                nn.Linear(200, 300),
                nn.Linear(300, 500),
                nn.Linear(500, 784)
            )

    def forward(self, X):
        X = X.view(-1, 784)
        code = self.encode(X)
        return self.decode(code), code

class AutoEncoder_DNN2(nn.Module):
    def __init__(self):
        super(AutoEncoder_DNN2, self).__init__()
        # TODO: Maybe need remove bias?
        self.encode = nn.Sequential(
                nn.Linear(784, 392),
                nn.ReLU(inplace=True),
                nn.Linear(392, 196),
                nn.ReLU(inplace=True),
                nn.Linear(196, 32),
                nn.ReLU(inplace=True),
            )
        self.decode = nn.Sequential(
                nn.Linear(32, 196),
                nn.ReLU(inplace=True),
                nn.Linear(196, 392),
                nn.ReLU(inplace=True),
                nn.Linear(392, 784),
            )
        for m in self.modules():
            if type(m) == nn.Linear:
                limit = np.sqrt(6 / (m.in_features + m.out_features))
                m.weight.data.uniform_(-limit, limit)
                m.bias.data.zero_()

    def forward(self, X):
        X = X.view(-1, 784)
        code = self.encode(X)
        return self.decode(code), code


