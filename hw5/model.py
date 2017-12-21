from import_modules import *
from util import *
class RatingClassifier():
    def __init__(self, config):
        self.config = config

    def split(self, train):
        try:
            with open("./train.pickle", "rb") as f1, open("./valid.pickle", "rb") as f2:
                print("Using pre-split")
                newtrain = pickle.load(f1)
                valid = pickle.load(f2)

        except FileNotFoundError:
            newtrain = {"U":train["U"], "M":train["M"], "userid2index":train["userid2index"],
            "movies2index": train["movies2index"]}
            valid = {"U":train["U"], "M":train["M"], "userid2index":train["userid2index"],
            "movies2index": train["movies2index"]}
            #
            indices = np.random.permutation(train["X"].shape[0])
            point = train["X"].shape[0] * 9 // 10
            # TODO:
            newtrain["X"] = train["X"][indices[:point]]
            newtrain["y"] = train["y"][indices[:point]]
            valid["X"] = train["X"][indices[point:]]
            valid["y"] = train["y"][indices[point:]]
            with open("./train.pickle", "wb") as f1, open("./valid.pickle", "wb") as f2:
                pickle.dump(newtrain, f1)
                pickle.dump(valid, f2)

        return newtrain, valid

    def build_model(self, train):
        augment_one_hot_embedding = None
        if AUGMENT:
            augment_one_hot_embedding = \
                movie_genre_one_hot("./data/movies.csv", train["M"], train["movies2index"])

        self.model = MF(6040, 3706, self.config.latent_dim, self.offset, 
            self.denominator, self.config.bias, augment_one_hot=augment_one_hot_embedding)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            #self.model.user_embedding.cpu()
            #self.model.movies_embedding.cpu()
        self.optimizer = optim.Adam(filter(lambda p:p.requires_grad, self.model.parameters()), 
            lr=self.config.lr)
        self.loss_fn = torch.nn.MSELoss()

    def predict_on_batch(self, X):
        #pdb.set_trace()
        self.model.eval()
        y_ = self.model.forward(X)
        self.model.train()
        y_ = y_.cpu().data.numpy()
        # restore normalization
        if NORMALIZE:
            y_  = y_ * self.denominator + self.offset
        y_ = np.clip(y_, 1.0, 5.0)
        return y_

    def predict(self, X):
        batch_size = self.config.batch_size
        g = batch_generator(X, None, batch_size, shuffle=False, training=False)
        N =  X.shape[0]
        quotient = N // batch_size
        remainder = N % batch_size
        y_ = []
        for batch in g:
            temp_y_ = self.predict_on_batch(batch["X"])
            y_.append(temp_y_)
        return np.concatenate(y_, axis=0)

    def train_on_batch(self, X, y, step=True):
        # zero grad
        self.optimizer.zero_grad()
        # forward
        y_ = self.model.forward(X)
        # count loss
        loss = self.loss_fn.forward(y_, y)
        
        if step:
            # backprop
            loss.backward()
            # clipping
            #_ = torch.nn.utils.clip_grad_norm(self.model.parameters(), 50)    
            # update parameter
            self.optimizer.step()
        # return loss and accuracy
        if NORMALIZE:
            return loss.data[0] * (self.denominator[0] **2) # restore to true MSE
        else:
            return loss.data[0]

    def run_epoch(self, train, valid):
        batch_size = self.config.batch_size
        # training
        g = batch_generator(train["X"], train["y"], self.config.batch_size, shuffle=True,
            training=True)
        #
        N = train["X"].shape[0]
        print("{} {}".format(train["X"].shape[0], valid["X"].shape[0]))
        quotient = N // batch_size
        remainder = N % batch_size
        train_losses = []
        for batch in g:
            loss = self.train_on_batch(batch["X"], batch["y"])
            #
            train_losses.append(loss * batch["X"].size()[0])
        train_loss = np.sqrt(np.sum(train_losses) / N)
        # validation
        g = batch_generator(valid["X"], valid["y"], self.config.batch_size, shuffle=False,
            training=False)
        #
        N = valid["X"].shape[0]
        quotient = N // batch_size
        remainder = N % batch_size
        valid_losses = []
        for batch in g:
            batch_y_ = self.predict_on_batch(batch["X"])
            batch_loss = np.sum(np.square(batch["y"].cpu().data.numpy() - batch_y_))
            #
            valid_losses.append(batch_loss)
        valid_loss = np.sqrt(np.sum(valid_losses) / N)
        return train_loss, valid_loss

    def fit(self, train, valid, now, load_model=True):
        #
        
        if NORMALIZE:
            if NORMALIZE_TYPE == "M":
                print("Using M")
                self.offset = np.array([min(train["y"])])
                self.denominator = np.array([max(train["y"]) - min(train["y"])])
            elif NORMALIZE_TYPE == "Z":
                print("Using Z")
                self.offset = np.array([np.mean(train["y"])])
                self.denominator = np.array([np.std(train["y"], ddof=1)])
            # normalize 
            train["y"] = (train["y"] - self.offset) / self.denominator
        else:
            self.offset = None
            self.denominator = None
        self.build_model(train)
        if load_model:
            self.model.load_state_dict(torch.load("./param/time: 20171216_2311"))
            return None, None, -1, None, None
        #
        training_accuracy_list = []
        valid_accuracy_list = []
        train_loss_list = []
        valid_loss_list = []
        min_valid_loss = float("inf")
        try:
            for epoch in range(self.config.n_epochs):
                train_loss, valid_loss = \
                    self.run_epoch(train, valid)
                print("Epochs {} out of {}".format(epoch+1, self.config.n_epochs))
                if valid_loss < min_valid_loss:
                    min_valid_loss = valid_loss
                    # save model
                    torch.save(self.model.state_dict(), "./param/"+str(now))

                train_loss_list.append(train_loss)
                valid_loss_list.append(valid_loss)
                print("Average Loss {}".format(train_loss))
                print("Min valid loss {}".format(min_valid_loss))
        except KeyboardInterrupt:
            pass
        self.model.load_state_dict(torch.load("./param/"+ str(now)))
        return training_accuracy_list, valid_accuracy_list, min_valid_loss, train_loss_list, \
        valid_loss_list

class MF(nn.Module):
    def __init__(self, users_count, movies_count, dimension, offset, denominator, bias=False, DNN=False,
        augment_one_hot=None):
        super(MF, self).__init__()
        self.user_embedding = nn.Embedding(users_count, dimension)
        self.user_embedding.weight.data.uniform_(-0.05, 0.05)
        self.movies_embedding = nn.Embedding(movies_count, dimension)
        self.movies_embedding.weight.data.uniform_(-0.05, 0.05)
        self.augment_one_hot = augment_one_hot
        if augment_one_hot is not None:
            self.movies_aug_embedding = nn.Embedding(movies_count, augment_one_hot.shape[1])
            # turn off gradient
            self.movies_aug_embedding.requires_grad = False
            # augment one hot encoding
            self.movies_aug_embedding.weight.data.copy_(torch.from_numpy(augment_one_hot))
            # bilinear
            self.bilinear = nn.Linear(dimension + augment_one_hot.shape[1], dimension, bias=False)
            self.bilinear_drop = nn.Dropout(p=0.4)

        if NORMALIZE:
            self.offset = nn.Parameter(torch.from_numpy(offset).float(), requires_grad=False)
            self.denominator = nn.Parameter(torch.from_numpy(denominator).float(), requires_grad=False)
        self.bias = bias
        self.DNN = DNN
        if DNN:
            self.linear = nn.Sequential(
                    nn.BatchNorm1d(2*dimension),
                    nn.Linear(2*dimension, 100),
                    nn.SELU(inplace=True),
                    nn.Dropout(p=0.2),
                    nn.BatchNorm1d(100),
                    nn.Linear(100, 50),
                    nn.SELU(inplace=True),
                    nn.Dropout(p=0.3),
                    nn.BatchNorm1d(50),
                    nn.Linear(50, 25),
                    nn.SELU(inplace=True),
                    nn.Dropout(p=0.4),
                    nn.BatchNorm1d(25),
                    nn.Linear(25, 1)
                )
            # initial with uniform
            for m in self.linear.modules():
                if isinstance(m, nn.Linear):
                    m.weight.data.uniform_(-0.1, 0.1)
        if bias:
            self.user_bias = nn.Embedding(users_count, 1)
            self.user_bias.weight.data.zero_()
            self.movies_bias = nn.Embedding(movies_count, 1)
            self.movies_bias.weight.data.zero_()
        self.drop = nn.Dropout(p=0.1)
        
        #pdb.set_trace()

    def forward(self, X):
        '''
            X = (batch_size, 2) (user_index and movie_index)
        '''
        # user index
        #
        users_indices = X[:, 0]
        movies_indices = X[:, 1]
        # (B, Dimension)
        users_matrix = self.user_embedding(users_indices)
        movies_matrix = self.movies_embedding(movies_indices)

        if self.augment_one_hot is not None:
            movies_aug_matrix = self.movies_aug_embedding(movies_indices)
            movies_matrix = torch.cat([movies_matrix, movies_aug_matrix], dim=1)
            # bilinear (B, D+A) -> (B, D)
            movies_matrix = self.bilinear_drop(self.bilinear(movies_matrix))


        if self.DNN:
            users_moives_matrix = torch.cat([users_matrix, movies_matrix], dim=1)
            y_ = self.linear(users_moives_matrix)
            # (B, 1) -> (B, )
            y_ = y_.squeeze(1)
            return y_
        # (B, D) * (B * D) -> (B)
        y_ = torch.sum(users_matrix * movies_matrix, dim=-1)
        if self.bias:
            b_user = self.user_bias(users_indices)
            b_movie = self.movies_bias(movies_indices)
            #pdb.set_trace()
            y_ = y_ + b_user.squeeze(-1) + b_movie.squeeze(-1)
        # normalize
        if NORMALIZE:
            y_ = (y_ - self.offset) / self.denominator

        return y_