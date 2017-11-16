from import_modules import *
from util import *
class BaseLineCNNWrapper():
    def __init__(self, config):
        self.config = config
        # construct model
        self.construct_model()
        '''
        if False:
            # visualize graph
            writer = SummaryWriter("graph_visualize")
            m = self.model
            res = m(Variable(torch.Tensor(1, 48, 48), requires_grad=True))
            writer.add_graph(m, res)

            writer.close()
            exit()
        '''
        

    def construct_model(self):
        if self.config.cuda == True:
            if self.config.option == "JimResidualNetwork":
                self.model = JimResidualNetwork().cuda(0)
            elif self.config.option == "DNN":
                self.model = DNN().cuda(0)
            elif self.config.option == "DeepResidualNetwork":
                self.model = DeepResidualNetwork(BasicBlock, [3, 3, 2, 2]).cuda(0)
        else:
            if self.config.option == "JimResidualNetwork":
                self.model = JimResidualNetwork()
        self.loss = torch.nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr, weight_decay=0.0001)

    def accuracy(self, y_, y):
        total_sum = np.sum((y_ == y))
        return total_sum / y_.shape[0]

    def argmax(self, fx):
        # reduce last dimension
        # pdb.set_trace()
        max_value, y_ = torch.max(fx, dim=-1)
        return y_

    def forward(self, inputs_batch, model, inference=False):
        if self.config.cuda == True:
            X = Variable(torch.from_numpy(inputs_batch).float(), 
            requires_grad=False, volatile=inference).cuda(0)
        else:
            X = Variable(torch.from_numpy(inputs_batch).float(), 
            requires_grad=False, volatile=inference)
        
        fx = model.forward(X)
        
        return fx

    def train_on_batch(self, inputs_batch, labels_batch, model, optimizer, loss_op):
        optimizer.zero_grad()
        # forward
        fx = self.forward(inputs_batch, model)
        if self.config.cuda == True:
            output = loss_op.forward(fx, Variable(torch.from_numpy(labels_batch).long(), 
            requires_grad=False).cuda(0))
        else:
            output = loss_op.forward(fx, Variable(torch.from_numpy(labels_batch).long(), 
            requires_grad=False))
        # backward
        output.backward()
        # Update parameters
        optimizer.step()
        
        return output.data[0]

    def run_epoch(self, train, model, optimizer, loss_op, batch_gener=None):
        # batch generator
        batch_size = self.config.batch_size
    
        if batch_gener is None:
            #
            batch_gener = batch_generator(train['X'], train['y'], self.config.batch_size, True)
            quotient = train['X'].shape[0] // batch_size
            remainder = train['X'].shape[0] - quotient * batch_size

            accuracy = 0.0
            for _ in range(quotient):
                batch = next(batch_gener)
                loss = self.train_on_batch(batch['X'], batch['y'], model, optimizer, loss_op)
                
                batch_y_ =  self.predict_batch(batch['X'], normalize=False)
                accuracy += self.accuracy(batch_y_, batch['y']) * batch_size
        else:
            
            batches = 0
            quotient = train['X'].shape[0] // batch_size
            accuracy = 0.0
            for tup, _ in zip(batch_gener.flow(np.expand_dims(train['X'],axis=-1), train['y'], batch_size=self.config.batch_size), 
                range(quotient)):
                batch_X, batch_y = tup[0], tup[1]

                #pdb.set_trace()
                loss = self.train_on_batch(batch_X, batch_y, model, optimizer, loss_op)
                
                batch_y_ = self.predict_batch(batch_X, normalize=False)
                accuracy += self.accuracy(batch_y_, batch_y) * batch_size
                batches += 1
                if batches == quotient:
                    break
        accuracy = accuracy / (quotient * batch_size)
        print("Last Loss {}".format(loss))
        return loss, accuracy 

    def fit(self, train, valid=None, now=None):
        # init logger
        logger = Logger(self.config.log)
        # split train valid
        if valid is None:
            # Note: train instance might been changed
            train, valid = self.split_train_valid(train)
            # dump to numpy file
            with open(os.path.join(self.config.param_folder, str(now)) + "train", 'wb') as f:
                pickle.dump(train, f)
            with open(os.path.join(self.config.param_folder, str(now)) + "valid", 'wb') as f:
                pickle.dump(valid, f)

        # if data aug enabled
        if self.config.data_aug == True:
            train = self.data_augmentation(train)
        
        print(train['X'].shape)
        # normalization
        train['X'] = self.normalization(train['X'])

        # keras batch generator
        batch_gener = ImageDataGenerator(
            rotation_range=10,
            #width_shift_range=0.2,
            #height_shift_range=0.2,
            zoom_range=[0.8, 1.2],
            #shear_range=0.2,
            horizontal_flip=True)

        batch_gener.fit(np.expand_dims(train['X'], axis=-1))
        losses = []
        while True:
            try:
                # training
                best_valid_accuracy = -1
                counter = 0
                #
                
                for epoch in range(self.config.n_epochs):
                    print("Epoch {} out of {}".format(epoch + 1, self.config.n_epochs))
                    loss, train_accuracy = self.run_epoch(train, self.model, self.optimizer, self.loss, 
                        batch_gener)
                    losses.append(loss)
                    
                    # training accuracy 
                    print("Training accuracy: {}".format(train_accuracy))
                    # validation accuracy
                    if valid is not None:
                        y_ = self.predict(valid['X'])
                        valid_accuracy = self.accuracy(y_, valid['y'])
                        print("Validation accuracy: {}".format(valid_accuracy))
                        
                        if valid_accuracy > best_valid_accuracy:
                            best_valid_accuracy = valid_accuracy
                            torch.save(self.model.state_dict(), "./param/"+ str(now))
                            counter = 0
                        counter += 1
                        print("Best valid accuracy: {}".format(best_valid_accuracy))

                    print('\nINFO:root:Epoch[%d] Train-accuracy=%f\nINFO:root:Epoch[%d] Validation-accuracy=%f' 
                        %(epoch, train_accuracy, epoch, valid_accuracy))
                    # record tensorboard
                    info = {"train_accuracy": train_accuracy, "valid_accuracy": valid_accuracy}
                    for tag, value in info.items():
                        logger.scalar_summary(tag, value, epoch)
                break
            except KeyboardInterrupt:
                res = input("If you want to re-take gradient step?")
                if res == "y" or res == "Y":
                    self.model.load_state_dict(torch.load(os.path.join(self.config.param_folder, str(now))))
                    continue
                else:
                    break
            
        # restore best
        self.model.load_state_dict(torch.load(os.path.join(self.config.param_folder, str(now))))
        print("Best valid accuracy: {}".format(best_valid_accuracy))

        return train_accuracy, best_valid_accuracy

    def predict_batch(self, X, normalize=True):
        self.model.eval()
        # squeeze
        if len(X.shape) == 4:
            X = np.squeeze(X, axis=-1)
        if normalize == True:
            X = (X - self.mean) / (self.std+1e-10)
        # recover dim
        X = np.expand_dims(X, axis=-1)

        fx = self.forward(X, self.model, inference=True)
        y_ = self.argmax(fx).data.cpu().numpy()
        # back to train
        self.model.train()
        return y_

    def predict(self, X):
        # divide it by batch
        y_ = []
        batch_gener = batch_generator(X, None, self.config.batch_size, False)
        for batch in batch_gener:
            batch_y_ = self.predict_batch(batch['X'])
            y_.append(batch_y_)
        y_ = np.concatenate(y_, axis=0)
        return y_


    def data_augmentation(self, train):
        factor = 2
        train_aug_X = [None] * (train['X'].shape[0] * factor)
        train_aug_y = [None] * (train['y'].shape[0] * factor)
        # add rotate 90
        for i, image in enumerate(train['X']):
            # rotate 45 degree
            
            # 360 / 30 = 12, 360 / 45 = 8
            # 1 2 3 4 .... 11 step, 1 2 3 .... 7
            train_aug_X[i*factor+0] = rotate(image, 10, reshape=False)
            train_aug_y[i*factor+0] = train['y'][i]
            train_aug_X[i*factor+1] = rotate(image, -10, reshape=False)
            train_aug_y[i*factor+1] = train['y'][i]
            # add value
            '''
            train_aug_X[i*factor+3] = np.clip(image + 25, 0, 255)
            train_aug_y[i*factor+3] = train['y'][i]
            # substract value
            train_aug_X[i*factor+4] = np.clip(image - 25, 0, 255)
            train_aug_y[i*factor+4] = train['y'][i]
            # multiply value
            train_aug_X[i*factor+5] = np.clip(image * 0.5, 0, 255)
            train_aug_y[i*factor+5] = train['y'][i]
            # multiply
            train_aug_X[i*factor+6] = np.clip(image * 1.5, 0, 255)
            train_aug_y[i*factor+6] = train['y'][i]
            '''
            # padding four direction
            '''
            train_aug_X[i*factor+7] = np.pad(image, [(2, 0), (0, 0)], mode="constant")[:-2, :]
            train_aug_y[i*factor+7] = train['y'][i]
            train_aug_X[i*factor+8] = np.pad(image, [(0, 2), (0, 0)], mode="constant")[2:, :]
            train_aug_y[i*factor+8] = train['y'][i]
            train_aug_X[i*factor+9] = np.pad(image, [(0, 0), (2, 0)], mode="constant")[:, :-2]
            train_aug_y[i*factor+9] = train['y'][i]
            train_aug_X[i*factor+10] = np.pad(image, [(0, 0), (0, 2)], mode="constant")[:, 2:]
            train_aug_y[i*factor+10] = train['y'][i]
            '''

        #pdb.set_trace()
        # concate aug
        aug_X = np.stack(train_aug_X, axis=0)
        aug_y = np.stack(train_aug_y, axis=0)

        # concate aug with original data
        train['X'] = np.concatenate([train['X'], aug_X], axis=0)
        train['y'] = np.concatenate([train['y'], aug_y], axis=0) 
        
        return train

    def split_train_valid(self, all_train):
        X = all_train['X']
        size = X.shape[0]
        indices = np.random.permutation(X.shape[0])
        # divid dataset to (train:valid) = (9:1) 
        point = (size * 8 ) // 10
        train = {}
        valid = {}
        train['X'] = np.take(all_train['X'], indices[:point], axis=0)
        train['y'] = np.take(all_train['y'], indices[:point], axis=0)
        valid['X'] = np.take(all_train['X'], indices[point:], axis=0)
        valid['y'] = np.take(all_train['y'], indices[point:], axis=0)
        return train, valid

    def normalization(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0, ddof=1)
        return (X - self.mean) / (self.std + 1e-10)

class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        # 
        self.layer1 = nn.Linear(48 * 48, 1800)
        self.layer2 = nn.Linear(1800, 1050)
        self.layer3 = nn.Linear(1050, 7)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X):
        X = torch.squeeze(X, -1)
        X = X.view(X.size()[0], 48 * 48)
        out = self.layer1(X)
        out = self.relu(out)
        out = self.layer2(out)
        out = self.relu(out)
        out = self.layer3(out)
        return out

class JimResidualNetwork(nn.Module):
    def __init__(self):
        super(JimResidualNetwork, self).__init__()
        # 48 -> 48 + 4 - 5 / 1 + 1 = 48
        self.layer1 = nn.Sequential(
            # 
            nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(1./20, inplace=True),
            nn.BatchNorm2d(64),
            # 48 -> 48 + 2 - 2 / 2 + 1 = 25
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.25)
            )
        self.layer2 = nn.Sequential(
            # 
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(1./20, inplace=True),
            nn.BatchNorm2d(128),
            # 25 + 1 - 2 / 2 + 1 = 13
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Dropout(0.3)
            )
        self.layer3 = nn.Sequential(
            # 13
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(1./20, inplace=True),
            nn.BatchNorm2d(256),
            # 13 + 1 - 2 / 2 + 1 = 7
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Dropout(0.35)
            )
        
        self.layer4 = nn.Sequential(
            # 
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(1./20, inplace=True),
            nn.BatchNorm2d(512),
            # 7 + 1 - 2 / 2 + 1 = 4
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Dropout(0.4)
            )
        
        #self.downsample = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Sequential(
            nn.Linear(512 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.5),

            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.5),

            nn.Linear(512, 7)
            )
        self.relu = nn.ReLU(inplace=True)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.uniform_(-math.sqrt(6. / n), math.sqrt(6. / n))
            elif isinstance(m, nn.Linear):
                n = m.in_features + m.out_features
                m.weight.data.uniform_(-math.sqrt(6. / n), math.sqrt(6. / n))
            #elif isinstance(m, nn.BatchNorm2d):
            #    m.weight.data.fill_(1)
            #    m.bias.data.zero_()

    def forward(self, X):
        # Simulate Resnet
        #pdb.set_trace()
        
        X = torch.squeeze(X, -1)
        X = torch.unsqueeze(X, 1)
        out = self.layer1(X)
        
        out = self.layer2(out)
        
        out = self.layer3(out)
        
        out = self.layer4(out)

        out = out.view(out.size()[0], -1)
        out = self.fc(out)
        
        return out

class BaseLineCNN(nn.Module):
    def __init__(self, in_channels=1, num_class=7):
        super(BaseLineCNN, self).__init__()
        # conv2d -> activation -> maxpool2d -> dropout -> 
        # -> flatten -> dense -> activation -> dense -> activation
        self.batch_norm = nn.BatchNorm2d(in_channels)
        self.conv = nn.Sequential(
            # 48 -> 46 -> 44
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=0),
            nn.Dropout(p=0.3),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=0),
            # 22
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),

            nn.Dropout(p=0.3),
            # 22 -> 20
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0),
            # 10
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),

            nn.Dropout(p=0.3),
            # 10 -> 8
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            # 4
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),

            nn.Dropout(p=0.3, inplace=False)
            )

        self.linears = nn.Sequential(
            nn.Linear(4 * 4 * 64, 4 * 4 * 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(4 * 4 * 64, num_class)
            )
    
    def forward(self, X):
        # unsqueeze channel into X
        #pdb.set_trace()
        out = torch.unsqueeze(X, 1)
        out = self.batch_norm(out)
        out = self.conv(out)
        out = out.view(-1, out.size()[1] * out.size()[2] * out.size()[3])
        out = self.linears(out)
        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # (C - 3 + 2) / 1 + 1
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.drop = nn.Dropout(p=0.25)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.drop(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        out = self.drop(out)
        return out

class BottleneckBlock(nn.Module):
    # channel dimension expansion
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BottleneckBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # increase stride size
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride,
            padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # expand dimension as the paper said
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.selu1 = nn.SELU(inplace=True)
        self.selu2 = nn.SELU(inplace=True)
        self.selu3 = nn.SELU(inplace=True)
        self.drop = nn.Dropout(p=0.2)
        self.downsample = downsample
        self.stride = stride

    def forward(self, X):
        residual = X
        #pdb.set_trace()
        out = self.conv1(X)
        out = self.bn1(out)
        out = self.selu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.selu2(out)
        out = self.drop(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            # Residual!!
            residual = self.downsample(X)

        out += residual
        out = self.selu3(out)
        return out


class DeepResidualNetwork(nn.Module):
    def __init__(self, block, layers, num_classes=7):
        super(DeepResidualNetwork, self).__init__()
        self.in_channels = 64
        # 48 -> (48 + 0 - 3) / 1 + 1 = 46
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, 
            padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # 46 -> 46 + 0 - 3 / 2 + 1 = 22
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, 
            padding=0)
        # if stride == 1 => 22 -> 22
        self.layer1 = self.make_layer(block, 64, layers[0], stride=1)
        # 
        self.layer2 = self.make_layer(block, 128, layers[1], stride=1)
        #
        #self.layer3 = self.make_layer(block, 256, layers[2], stride=1)
        #
        #self.layer4 = self.make_layer(block, 512, layers[3], stride=1)
        
        self.conv2 = nn.Conv2d(32, 512, kernel_size=3, stride=1, padding=1, bias=True)
        # 22 -> 22 + 0 - 3 / 1 + 1
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=1)
        self.fc = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(128 * block.expansion * 20 * 20, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            
            nn.Dropout(p=0.4),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            
            nn.Dropout(p=0.4),
            nn.Linear(512, num_classes)
            )
        self.drop = nn.Dropout(p=0.3)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def make_layer(self, block, channels, blocks, stride=1):
        downsample = None
        #pdb.set_trace()
        # in order to fix dimension for residual adding
        # if stride is not 1, we need to fix its image(feature map) size
        if stride != 1 or self.in_channels != channels * block.expansion:
            downsample = nn.Sequential(
                # keep same size with bottle neck
                nn.Conv2d(self.in_channels, channels * block.expansion,
                    kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channels * block.expansion)
                )

        layers = []
        layers.append(block(self.in_channels, channels, stride, downsample))
        self.in_channels = channels * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_channels, channels))

        return nn.Sequential(*layers)

    def forward(self, X):
        X = torch.squeeze(X, -1)
        X = torch.unsqueeze(X, 1)
        X = self.conv1(X)
        X = self.bn1(X)
        X = self.relu(X)
        X = self.maxpool(X)
        X = self.drop(X)

        X = self.layer1(X)
        X = self.layer2(X)
        #X = self.layer3(X)
        #X = self.layer4(X)

        X = self.maxpool2(X)
        X = X.view(X.size(0), -1)
        X = self.fc(X)

        return X

if __name__ == '__main__':
    DeepResidualNetwork(BottleneckBlock, [2, 2, 2, 2])