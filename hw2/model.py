from import_modules import *
from util import batch_generator
class LogisticRegression(object):
    def generate_param(self, X):
        self.w = np.zeros((X.shape[1]))
        self.b = np.zeros((1,))
    
    def feature_preprocessing(self, X):
        if self.config.option == "all":
            pass
        elif self.config.option == "0_1_3_4_5_power3":
            X = np.concatenate([X[:, [0,1,3,4,5]]**3 , X[:, [0,1,3,4,5]]**2, X], axis=1)
        elif self.config.option == "0_1_3_4_5_power2":
            X = np.concatenate([ X[:, [0,1,3,4,5]]**2, X], axis=1)
        else:
            raise NotImplementedError

        if self.config.normalize == "Z":
            if self.mu is None and self.sd is None:
                N = X.shape[0]
                self.mu = np.sum(X, axis=0) / N
                self.sd = np.sqrt(np.sum(np.square(X - self.mu), axis=0) / (N-1))
            X = (X - self.mu) / self.sd
        elif self.config.normalize == "no":
            pass
        else:
            raise NotImplementedError
        return X

    def forward_backward_prop(self, X, y):
        '''
        '''
        if self.w is None and self.b is None:
            self.generate_param(X)

        # forward pass
        z = np.dot(X, self.w) + self.b
        
        # sigmoid
        y_ = 1 / (1 + np.exp(-z))
        
        # backward prop
        N = X.shape[0]
        err = y - y_
        loss = -np.sum(y * np.log(y_+1e-15) + (1 - y) * np.log(1 - y_+1e-15)) / N
        grad_w = -np.dot(X.T, err) / N + 2 * self.config.regularize * self.w
        grad_b = -np.sum(err) / N
        return grad_w, grad_b, loss, y_

    def adagrad_w(self, grad):
        # adding square grad
        self.accumu_grad_w += np.square(grad)
        grad = grad / np.sqrt(self.accumu_grad_w)
        return grad

    def adagrad_b(self, grad):
        # adding square grad
        self.accumu_grad_b += np.square(grad)
        grad = grad / np.sqrt(self.accumu_grad_b)
        return grad
    
    def minimize(self, grad_w, grad_b):
        grad_w = self.adagrad_w(grad_w)
        grad_b = self.adagrad_b(grad_b)
        self.w = self.w - self.config.lr * grad_w
        self.b = self.b - self.config.lr * grad_b

    def accuracy(self, y, y_):
        print(y.shape)
        print(y_.shape)
        accu = np.mean((y == y_))
        return accu

    def run_epoch(self, train):
        losses = []
        # batch generator
        batch_gener = batch_generator(train['X'], train['y'], self.config.batch_size, True)
        
        for batch in batch_gener:
            loss = self.train_on_batch(batch['X'], batch['y'])
            losses.append(loss)
        
        #print("Last Loss {}".format(loss))
        return losses

    def train_on_batch(self, inputs_batch, labels_batch):
        X = self.feature_preprocessing(inputs_batch)
        grad_w, grad_b, loss, y_ = self.forward_backward_prop(X, labels_batch)
        self.minimize(grad_w, grad_b)
        return loss

    def fit(self, train, valid=None):
        # generate mu and sd
        self.feature_preprocessing(train['X'])
        # training
        losses = []
        for epoch in range(self.config.n_epochs):
            #print("Epoch {} out of {}".format(epoch + 1, self.config.n_epochs))
            loss = self.run_epoch(train)
            losses.append(loss)
    
        # accuracy
        train['y_'] = self.predict(train['X'])
        print("Train accuracy {}".format(str(self.accuracy(train['y'], train['y_']))))
        if valid is not None:
            valid['y_'] = self.predict(valid['X'])
            print("Valid accuracy {}".format(str(self.accuracy(valid['y'], valid['y_']))))

        
        # total training set error
        _, _, train_loss, _ = self.forward_backward_prop(self.feature_preprocessing(train['X']), train['y'])
        valid_loss = None
        if valid is not None:
            # validation set error
            _, _, valid_loss, _ = self.forward_backward_prop(self.feature_preprocessing(valid['X']), valid['y'])
        print(self.w)
        return train_loss, valid_loss
  
    def predict(self, X):
        if self.config.load_prev:
            self.load_param()
        X = self.feature_preprocessing(X)
        z = np.dot(X, self.w) + self.b
        y_ = 1 / (1 + np.exp(-z))
        y_ = (y_ >= 0.5).astype(int)
        return y_

    def load_param(self):
        timestamp = self.config.timestamp
        param = ['w', 'b', 'mu', 'sigma']
        filename = ["./param/%s_%s" %  (p, timestamp) for p in param]
        self.w = np.load(filename[0])
        self.b = np.load(filename[1])
        self.mu = np.load(filename[2])
        self.sd = np.load(filename[3])

    def save_param(self, filename):
        self.w.dump(filename[0])
        self.b.dump(filename[1])
        if self.mu is not None:
            self.mu.dump(filename[2])
        if self.sd is not None:
            self.sd.dump(filename[3])
    
    def build(self):
        self.w = None
        self.b = None
        self.mu = None
        self.sd = None
        self.accumu_grad_w = 1e-15
        self.accumu_grad_b = 1e-15

    def __init__(self, config):
        self.config = config
        self.build()

class GenerativeModel():
    def __init__(self, config):
        self.config = config
        self.build()

    def accuracy(self, y, y_):
        print(y.shape)
        print(y_.shape)
        accu = np.mean((y == y_))
        return accu

    def build(self):
        self.mu = self.sd = None

    def feature_preprocessing(self, X):
        if self.config.option == "all":
            pass
        elif self.config.option == "0_1_3_4_5_power2":
            X = np.concatenate([ X[:, [0,1,3,4,5]]**2, X], axis=1)
        else:
            raise NotImplementedError
        if self.config.normalize == "Z":
            if self.mu is None and self.sd is None:
                N = X.shape[0]
                self.mu = np.sum(X, axis=0) / N
                self.sd = np.sqrt(np.sum(np.square(X - self.mu), axis=0) / (N-1))
            X = (X - self.mu) / self.sd
        elif self.config.normalize == "no":
            pass
        else:
            raise NotImplementedError
        return X

    def fit(self, train, valid=None):
        # Count class 0
        N = train['y'].shape[0]
        X = self.feature_preprocessing(train['X'])
        # Group class 0 and 1
        X_class_zero = X[[train['y'] == 0]]
        X_class_one = X[[train['y'] == 1]]
        assert X_class_one.shape[0] + X_class_zero.shape[0] == train['X'].shape[0]
        # Count Probability
        N1  = X_class_zero.shape[0]
        N2 = X_class_one.shape[0]
        self.P_zero = X_class_zero.shape[0] / N
        self.P_one = X_class_one.shape[0] / N 
        self.mu_zero = np.mean(X_class_zero, axis=0)
        self.mu_one = np.mean(X_class_one, axis=0)
        self.covariance_zero = 0.0
        self.covariance_one = 0.0

        for vector in X_class_zero:
            diff = vector - self.mu_zero
            product = np.dot(np.reshape(diff, [-1, 1]), np.reshape(diff, [1, -1]))
            self.covariance_zero += product

        for vector in X_class_one:
            diff = vector - self.mu_one
            product = np.dot(np.reshape(diff, [-1, 1]), np.reshape(diff, [1, -1]))
            self.covariance_one += product
        assert np.array_equal(self.covariance_zero.T, self.covariance_zero)
        assert np.array_equal(self.covariance_one.T, self.covariance_one)
        
        self.covariance_zero = self.covariance_zero / (X_class_zero.shape[0])
        self.covariance_one = self.covariance_one / (X_class_one.shape[0])
        
        self.covariance = self.P_zero * self.covariance_zero + self.P_one * self.covariance_one
        # generate weight
        cov_inverse = np.linalg.inv(self.covariance)
        self.w = np.dot((self.mu_zero - self.mu_one), cov_inverse)
        self.b = (-1/2) * (self.mu_zero).dot(cov_inverse).dot(self.mu_zero) + \
        (1/2) * self.mu_one.dot(cov_inverse).dot(self.mu_one) + \
        np.log(N1/N2)

        return "GenerativeModel has no loss", "GenerativeModel has no loss"

    def predict(self, X):
        X = self.feature_preprocessing(X)
        print(X.shape)
        print(self.w.shape)
        print(self.b.shape)
        y_ = np.dot(X, self.w) + self.b
        y_ = 1 / (1 + np.exp(-y_))
        y_ = (y_ < 0.5)
        return y_
'''
class Torch_Logistic_Regression():
    def generate_param(self, X):
        input_dim = X.shape[1]
        model = torch.nn.Sequential()
        model.add_module("linear_1",
                     torch.nn.Linear(input_dim, 300, bias=True))
        model.add_module("relu1", torch.nn.ELU())
        model.add_module("dropout_1", torch.nn.Dropout(0.2))

        model.add_module("linear_2",
                     torch.nn.Linear(300, 200, bias=True))
        model.add_module("relu2", torch.nn.ELU())
        model.add_module("dropout_2", torch.nn.Dropout(0.2))
        
        model.add_module("linear_3", 
                     torch.nn.Linear(200, 1, bias=True))

        model.add_module("sigmoid", torch.nn.Sigmoid())
        loss = torch.nn.BCELoss(size_average=True)
        optimizer = optim.Adam(model.parameters(), lr=self.config.lr)
        return model, loss, optimizer

    def feature_preprocessing(self, X):
        if self.config.option == "all":
            pass
        elif self.config.option == "0_1_3_4_5_power2":
            X = np.concatenate([ X[:, [0,1,3,4,5]]**2, X], axis=1)
        else:
            raise NotImplementedError
        if self.config.normalize == "Z":
            if self.mu is None and self.sd is None:
                N = X.shape[0]
                self.mu = np.sum(X, axis=0) / N
                self.sd = np.sqrt(np.sum(np.square(X - self.mu), axis=0) / (N-1))
            X = (X - self.mu) / self.sd
        elif self.config.normalize == "no":
            pass
        else:
            raise NotImplementedError
        return X

    def accuracy(self, y, y_):
        print(y.shape)
        print(y_.shape)
        accu = np.mean((y == y_))
        return accu

    def predict(self, X):
        model = self.model
        model.eval()
        X = self.feature_preprocessing(X)
        X = torch.from_numpy(X).float()
        x = Variable(X, requires_grad=False)
        output = model.forward(x).view(-1)
        model.train()
        return (output.data.numpy() > 0.5).astype(int)

    def forward_backward_prop(self, x_val, y_val, model, optimizer, loss):
        x_val = torch.from_numpy(x_val).float()
        y_val = torch.from_numpy(y_val).float()
        
        X = Variable(x_val, requires_grad=False)
        y = Variable(y_val, requires_grad=False)
        
        # Reset gradient
        optimizer.zero_grad()

        # Foward
        fx = model.forward(X)
        output = loss.forward(fx.view(-1), y)
        #print(y.size())
        #print(fx.view(-1).size())
        # Backward
        output.backward()
        return output.data[0]

    def train_on_batch(self, inputs_batch, labels_batch, model, optimizer, loss_op):
        X = self.feature_preprocessing(inputs_batch)
        loss = self.forward_backward_prop(X, labels_batch, model, optimizer, loss_op)

        # Update parameters
        optimizer.step()
        
        return loss

    def run_epoch(self, train, model, optimizer, loss_op):
        losses = []
        # batch generator
        batch_gener = batch_generator(train['X'], train['y'], self.config.batch_size, True)
        
        for batch in batch_gener:
            loss = self.train_on_batch(batch['X'], batch['y'], model, optimizer, loss_op)
            losses.append(loss)
        
        #print("Last Loss {}".format(loss))
        return losses

    def fit(self, train, valid=None):
        # generate mu and sd
        X = self.feature_preprocessing(train['X'])
        # generate model
        self.model, self.loss_op, self.optimizer = self.generate_param(X)

        # training
        losses = []
        for epoch in tqdm(range(self.config.n_epochs)):
            #print("Epoch {} out of {}".format(epoch + 1, self.config.n_epochs))
            loss = self.run_epoch(train, self.model, self.optimizer, self.loss_op)
            losses.append(loss)
    
        # accuracy
        train['y_'] = self.predict(train['X'])
        print("Train accuracy {}".format(str(self.accuracy(train['y'], train['y_']))))
        if valid is not None:
            valid['y_'] = self.predict(valid['X'])
            print("Valid accuracy {}".format(str(self.accuracy(valid['y'], valid['y_']))))

        
        # total training set error
        train_loss = self.forward_backward_prop(self.feature_preprocessing(train['X']), train['y'], 
            self.model, self.optimizer, self.loss_op)
        valid_loss = None
        if valid is not None:
            # validation set error
            valid_loss= self.forward_backward_prop(self.feature_preprocessing(valid['X']), valid['y'],
                self.model, self.optimizer, self.loss_op)
        return train_loss, valid_loss


    def load_param(self):
        timestamp = self.config.timestamp
        param = ['w', 'b', 'mu', 'sigma']
        filename = ["./param/%s_%s" %  (p, timestamp) for p in param]
        self.model = torch.load(filename[0])
        self.mu = np.load(filename[2])
        self.sd = np.load(filename[3])

    def save_param(self, filename):
        torch.save(self.model, filename[0])
        if self.mu is not None:
            self.mu.dump(filename[2])
        if self.sd is not None:
            self.sd.dump(filename[3])

    def build(self):
        self.mu = self.sd = None

    def __init__(self, config):
        #torch.cuda.set_device(0)
        self.config = config
        self.build()
'''
class RandomForestModel():
    def feature_preprocessing(self, X):
        if self.config.option == "all":
            pass
        elif self.config.option == "0_1_3_4_5_power2":
            X = np.concatenate([ X[:, [0,1,3,4,5]]**2, X], axis=1)
        else:
            raise NotImplementedError
        if self.config.normalize == "Z":
            if self.mu is None and self.sd is None:
                N = X.shape[0]
                self.mu = np.sum(X, axis=0) / N
                self.sd = np.sqrt(np.sum(np.square(X - self.mu), axis=0) / (N-1))
            X = (X - self.mu) / self.sd
        elif self.config.normalize == "no":
            pass
        else:
            raise NotImplementedError
        return X
    def fit(self, train, valid=None):
        X = self.feature_preprocessing(train['X'])
        self.model.fit(X, train['y'])
        return "No", "No"

    def predict(self, X):
        X = self.feature_preprocessing(X)
        y_ = self.model.predict(X)
        return y_

    def accuracy(self, y, y_):
        print(y.shape)
        print(y_.shape)
        accu = np.mean((y == y_))
        return accu

    def load_param(self):
        timestamp = self.config.timestamp
        param = ['w', 'b', 'mu', 'sigma']
        filename = ["./param/%s_%s" %  (p, timestamp) for p in param]
        with open(filename[0], "rb") as f:
            self.model = pickle.load(f)
        self.mu = np.load(filename[2])
        self.sd = np.load(filename[3])

    def save_param(self, filename):
        
        with open(filename[0], "wb") as f:
            pickle.dump(self.model, f)
        if self.mu is not None:
            self.mu.dump(filename[2])
        if self.sd is not None:
            self.sd.dump(filename[3])

    def build(self):
        #best
        self.model = XGBClassifier(learning_rate=0.01, n_estimators=5000)
        self.mu = self.sd = None

    def __init__(self, config):
        self.config = config
        self.build()
