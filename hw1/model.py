import numpy as np
from tqdm import tqdm
class LinearRegressionModel(object):
    def generate_param(self):
        self.w = np.zeros((self.config.feature))
        self.b = np.zeros((1,))

    def forward_backward_prop(self, X, y):
        '''
        X is (batch_size, feature)
        w is (feature, )
        y is (batch_size, )
        gradient:
            grad = (-2 / n) * (y - y_).T \dot X
        '''
        # TODO: feature choosing
        X = np.reshape(X, (X.shape[0], -1))
        # forward
        y_ = np.dot(X, self.w) + self.b
        # backward: derivative of mean square error
        grad_w = (-2 / X.shape[0]) * np.dot((y - y_.T), X)
        grad_b = (-2 / X.shape[0]) * np.sum(y - y_, axis=0)
        # loss
        loss = np.sqrt(np.sum(np.square(y_ - y)) / X.shape[0])
        #print(grad.shape)
        return grad_w, grad_b, loss

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
        grad_w = self.adagrad_w(grad_w)
        grad_b = self.adagrad_b(grad_b)
        self.w = self.w - self.config.lr * grad_w
        self.b = self.b - self.config.lr * grad_b
    
    def train_on_batch(self, inputs_batch, labels_batch):
        grad_w, grad_b ,loss = self.forward_backward_prop(inputs_batch, labels_batch)
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
        losses = []
        for epoch in range(self.config.n_epochs):
            print("Epoch {} out of {}".format(epoch + 1, self.config.n_epochs))
            loss = self.run_epoch(train)
            losses.append(loss)
        # validation set error
        valid_loss = self.predict(valid)
        return valid_loss

    def build(self):
        # generate weight w
        self.generate_param()
        # accumulative gradient sum
        self.accumu_grad_w = 0.0
        self.accumu_grad_b = 0.0

    def predict(self, valid):
        _, _, loss = self.forward_backward_prop(valid['X'], valid['y'])
        return loss
    def __init__(self, config):
        self.config = config
        # build
        self.build()
