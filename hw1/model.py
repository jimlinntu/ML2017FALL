import numpy as np
#from tqdm import tqdm
class LinearRegressionModel(object):
    def generate_param(self, X):
        self.w = np.ones((X.shape[1]))
        self.b = np.zeros((1,))

    def feature_preprocessing(self, X):
        # feature choosing
        PM_index = self.metric_names.index("PM2.5")
        NO_index = self.metric_names.index("NO")
        PM10_index = self.metric_names.index("PM10")
        O3_index = self.metric_names.index("O3")
        if self.option == "ALL":
            X = np.reshape(X, (X.shape[0], -1))
        elif self.option == "PM2.5_9":
            X = X[:, PM_index, :]
        elif self.option == "PM2.5_5":
            X = X[:, PM_index, 4:]
        elif self.option == "PM2.5_5_power2":
            # X = (batch_size, feature) X^2 = (batch_size, feature) concate
            X = np.concatenate([X[:, PM_index, 4:], np.square(X[:, PM_index, 4:])], axis=1)
        elif self.option == "PM2.5_power2":
            # X = (batch_size, feature) X^2 = (batch_size, feature) concate
            X = np.concatenate([X[:, PM_index, :], np.square(X[:, PM_index, :])], axis=1)
        elif self.option == "PM2.5_PM10":
            # X = (batch_size, feature) X^2 = (batch_size, feature) concate
            X = np.concatenate([X[:, PM_index, :], X[:, PM10_index, :]], axis=1)
        elif self.option == "PM2.5_power2_PM10":
            # X = (batch_size, feature) X^2 = (batch_size, feature) concate
            X = np.concatenate([X[:, PM_index, :], X[:, PM_index, :] **2, 
                X[:, PM10_index, :]], axis=1)
        elif self.option == "PM2.5_power2_PM10_power2":
            # X = (batch_size, feature) X^2 = (batch_size, feature) concate
            X = np.concatenate([X[:, PM_index, :], X[:, PM_index, :] **2, 
                X[:, PM10_index, :], X[:, PM10_index] ** 2], axis=1)
        elif self.option == "PM2.5_power3":
            X = np.concatenate([X[:, PM_index, :], np.square(X[:, PM_index, :]), 
                X[:, PM_index, :] ** 3], axis=1)
        elif self.option == "PM2.5_power2_NO":
            X = np.concatenate([X[:, PM_index, :], np.square(X[:, PM_index, :]), 
                X[:, NO_index, :]], axis=1)
        elif self.option == "PM2.5_power2_NO_PM10":
            X = np.concatenate([X[:, PM_index, :], np.square(X[:, PM_index, :]), 
                X[:, NO_index, :], X[:, PM10_index, :]], axis=1)
        elif self.option == "PM2.5_power2_NO_InverseO3":
            X = np.concatenate([X[:, PM_index, :], np.square(X[:, PM_index, :]), 
                X[:, NO_index, :], -X[:, O3_index, :]], axis=1)
        elif self.option == "PM2.5_power2_NO_power2":
            X = np.concatenate([X[:, PM_index, :], np.square(X[:, PM_index, :]), 
                X[:, NO_index, :], np.square(X[:, NO_index, :])], axis=1)
        elif self.option == "PM2.5_power3_NO":
            X = np.concatenate([X[:, PM_index, :], np.square(X[:, PM_index, :]), 
                X[:, PM_index, :] ** 3, X[:, NO_index, :]], axis=1)
        elif self.option == "PM2.5_power2_NO_power3":
            X = np.concatenate([X[:, PM_index, :], np.square(X[:, PM_index, :]), 
                X[:, NO_index, :], np.square(X[:, NO_index, :]), X[:, NO_index, :] ** 3], axis=1)
        else:
            raise NotImplementedError 
        # normalize it
        # generate mu and sd
        if self.config.normalize == "Z":
            if self.mu is None and self.sd is None:
                N = X.shape[0]
                self.mu = np.sum(X, axis=0) / N
                self.sd = np.sqrt(np.sum(np.square(X - self.mu), axis=0) / (N-1))
            X = (X - self.mu) / self.sd 
        elif self.config.normalize == "minmax":
            if self.min is None and self.max is None:
                N = X.shape[0]
                self.min = np.amin(X, axis=0)
                self.max = np.amax(X, axis=0)
            X = (X - self.min) / (self.max - self.min) 
        else:
            raise NotImplementedError 
        return X

    def forward_backward_prop(self, X, y):
        '''
        X is (batch_size, feature)
        w is (feature, )
        y is (batch_size, )
        gradient:
            grad = (-2 / n) * (y - y_).T \dot X + 2 * lambda * w
        '''
        # forward
        if self.w is None and self.b is None:
            self.generate_param(X)
        grad_w, grad_b, loss, y_ = None, None, None, None
        y_ = np.dot(X, self.w) + self.b
        if y is not None:
            # backward: derivative of mean square error
            grad_w = (-2 / X.shape[0]) * np.dot(X.T, (y - y_))
            if self.config.regularize is not None:
                grad_w += 2 * self.config.regularize * self.w
            grad_b = (-2 / X.shape[0]) * np.sum((y - y_), axis=0)
            # loss
            loss = np.sqrt(np.sum(np.square(y_ - y)) / X.shape[0])
        
        return grad_w, grad_b, loss, y_

    def adam_opt(self, grad_w, grad_b):
        self.t += 1
        self.moment_w = self.beta1 * self.moment_w + (1 - self.beta1) * grad_w
        self.v_w = self.beta2 * self.v_w + (1 - self.beta2) * (grad_w ** 2)
        m_hat = self.moment_w / (1 - self.beta1 ** self.t)
        v_hat = self.v_w / (1 - self.beta2 ** self.t)
        grad_w = m_hat / (np.sqrt(v_hat) + self.epsilon)

        self.moment_b = self.beta1 * self.moment_b + (1 - self.beta1) * grad_b
        self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * (grad_b ** 2)
        m_hat = self.moment_b / (1 - self.beta1 ** self.t)
        v_hat = self.v_b / (1 - self.beta2 ** self.t)
        grad_b = m_hat / (np.sqrt(v_hat) + self.epsilon)
        return grad_w, grad_b

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
        # gradient descent
        grad_w, grad_b = self.adam_opt(grad_w, grad_b)
        self.w = self.w - self.config.lr * grad_w
        self.b = self.b - self.config.lr * grad_b
    
    def train_on_batch(self, inputs_batch, labels_batch):
        X = self.feature_preprocessing(inputs_batch)
        grad_w, grad_b ,loss, _ = self.forward_backward_prop(X, labels_batch)
        self.minimize(grad_w, grad_b)
        return loss

    def run_epoch(self, train):
        losses = []
        # batch generator
        batch_generator = self.config.loader.batch_generator(train['X'], train['y'])
        N_example = train['X'].shape[0]
        length = N_example // self.config.batch_size

        for i in range(length):
            batch = next(batch_generator)
            loss = self.train_on_batch(batch['X'], batch['y'])
            losses.append(loss)
            
        print("Last Loss {}".format(loss))
        return losses

    def fit(self, train, valid):
        # generate mu and sd
        self.feature_preprocessing(train['X'])
        # training
        losses = []
        for epoch in range(self.config.n_epochs):
            print("Epoch {} out of {}".format(epoch + 1, self.config.n_epochs))
            loss = self.run_epoch(train)
            losses.append(loss)
       
        # total training set error
        _, _, train_loss, _ = self.forward_backward_prop(self.feature_preprocessing(train['X']), train['y'])
        valid_loss = None
        if valid is not None:
            # validation set error
            _, _, valid_loss, _ = self.forward_backward_prop(self.feature_preprocessing(valid['X']), valid['y'])
        return train_loss, valid_loss

    def build(self):
        # generate weight w
        self.min = self.max = None
        self.w, self.b, self.mu, self.sd = None, None, None, None
        # accumulative gradient sum
        self.t = 0
        self.v_w, self.v_b = 0.0, 0.0
        self.moment_w = 0.0
        self.moment_b = 0.0
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.accumu_grad_w = 0.0
        self.accumu_grad_b = 0.0
        self.metric_names = ['AMB_TEMP', 'CH4', 'CO', 'NMHC', 
        'NO', 'NO2', 'NOx', 'O3', 'PM10', 'PM2.5', 'RAINFALL', 
        'RH', 'SO2', 'THC', 'WD_HR', 'WIND_DIREC', 'WIND_SPEED', 'WS_HR']
        # option
        self.option = self.config.option

    def predict(self, test):
        if self.config.load_prev:
            self.load_param()
        X = self.feature_preprocessing(test['X'])
        y_ = np.dot(X, self.w) + self.b
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
            self.sd.dump(filename[3])

    def __init__(self, config):
        self.config = config
        # build
        self.build()
