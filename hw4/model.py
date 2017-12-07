from import_modules import *
from util import *

class SentimentClassifier():
    def __init__(self, config):
        self.config = config

    def build_model(self):
        if not USE_BOW:
            self.model = DeepJimNetwork(self.voc.n_words, 100, 200, 1, True, self.voc.embedding_matrix)
        else:
            self.model = DNN(self.bow.dim)
            print(self.model)

        if USE_CUDA:
            self.model = self.model.cuda()

        self.optimizer = optim.RMSprop(filter(lambda p:p.requires_grad, self.model.parameters()), \
            lr=self.config.lr)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def run_epoch(self, train_df, valid_df, sample_encode):
        # 
        train_losses = []

        # generator
        if USE_BOW != True:
            generator = sample_encode.generator(train_df, self.config.batch_size)
        else:
            generator = self.bow.batch_generator(train_df, self.config.batch_size)

        N = train_df.shape[0]
        q = train_df.shape[0] // self.config.batch_size + int(train_df.shape[0] % self.config.batch_size !=0)
        training_accuracy = 0.
        
        for batch in tqdm(generator, total=q):
            #
            self.optimizer.zero_grad()
            #
            if USE_CUDA:
                batch['X'] = batch['X'].cuda()
                batch['y'] = batch['y'].cuda()
                

            # forward
            y_ = self.model.forward(batch['X'], batch['lengths'])

            # count loss
            loss = self.loss_fn.forward(y_, batch['y'])

            # backprop
            loss.backward()
           
            # clip gradient
            _ = torch.nn.utils.clip_grad_norm(self.model.parameters(), 50.0)

            # update parameter
            self.optimizer.step()

            # append loss
            train_losses.append(loss.data[0])

            # test training data accuracy

            training_accuracy += self.accuracy(np.argmax(y_.data.cpu().numpy(), axis=1), \
                batch['y'].data.cpu().numpy())

        training_accuracy /= N
        print("Training accuracy: {}".format(training_accuracy))
        # test validation 
        N = valid_df.shape[0]
        q = valid_df.shape[0] // self.config.batch_size + int(valid_df.shape[0] % self.config.batch_size !=0)
        if not USE_BOW:
            generator = sample_encode.generator(valid_df, self.config.batch_size)
        else:
            generator = self.bow.batch_generator(valid_df, self.config.batch_size)
        valid_accuracy = 0.
        valid_losses = []
        for batch in tqdm(generator, total=q):
            self.model.eval()
            batch['X'] = batch['X'].cuda() if USE_CUDA else batch['X']
            batch['y'] = batch['y'].cuda() if USE_CUDA else batch['y']
            
            y_, argmax_y_, loss = self.predict_on_batch(batch)
            
            valid_losses.append(loss)
            valid_accuracy += self.accuracy(argmax_y_, batch['y'].data.cpu().numpy())
            self.model.train()

        valid_accuracy /= N
        print("Validation accuracy: {}".format(valid_accuracy))
        # predict validation result
        return np.mean(train_losses), np.mean(valid_losses), training_accuracy, valid_accuracy

    def fit(self, train_df, valid_df, train_df_no_label, now):
        if self.config.load_model:
            # load voc
            with open("./param/voc", "rb") as f:
                self.voc = pickle.load(f)
            # build model
            self.build_model()
            # load model
            self.model.load_state_dict(torch.load("./param/"+ "time: 20171208_0003"))
            return None, None, None, None, -1
        
        
        if USE_BOW != True:
            # fit to voc
            try:
                with open("./param/voc", "rb") as f:
                    self.voc = pickle.load(f)
                    
            except FileNotFoundError:
                self.voc = Voc()
                self.voc.fit(train_df, train_df_no_label)
                with open("./param/voc", "wb") as f:
                    pickle.dump(self.voc, f)
        else:
            self.bow = BOW()
            # count bag of word
            self.bow.fit(train_df, train_df_no_label)



        # build model
        self.build_model()

        # build sample encode
        if USE_BOW != True:
            sample_encode = Sample_Encode(self.voc)
        else:
            sample_encode = None

        training_accuracy_list = []
        valid_accuracy_list = []
        train_loss_list = []
        valid_loss_list = []
        max_valid_accuracy = -1
        try:
            for epoch in range(self.config.n_epochs):
                print("Epochs {} out of {}".format(epoch+1, self.config.n_epochs))
                
                train_loss, valid_loss, training_accuracy, valid_accuracy = \
                self.run_epoch(train_df, valid_df, sample_encode)

                if valid_accuracy > max_valid_accuracy:
                    max_valid_accuracy = valid_accuracy
                    # save model
                    torch.save(self.model.state_dict(), "./param/"+str(now))

                training_accuracy_list.append(training_accuracy)
                valid_accuracy_list.append(valid_accuracy)
                train_loss_list.append(train_loss)
                valid_loss_list.append(valid_loss)
                print("Average Loss {}".format(train_loss))
                print("Max valid_accuracy: {}".format(max_valid_accuracy))
        except KeyboardInterrupt:
            pass
        self.model.load_state_dict(torch.load("./param/"+ str(now)))

        return training_accuracy_list, valid_accuracy_list, max_valid_accuracy, train_loss_list, \
        valid_loss_list

    def accuracy(self, y_, y):
        assert y_.shape == y.shape
        return np.sum((y_ == y))

    def predict_on_batch(self, batch):
        istrain = 'y' in batch and batch['y'] is not None
        self.model.eval()
        y_ = self.model.forward(batch['X'], batch['lengths'])
        if istrain:   
            loss = self.loss_fn.forward(y_, batch['y'])
        y_ = F.softmax(y_).data.cpu().numpy()
        argmax_y_ = np.argmax(y_, axis=1)
        self.model.train()
        if istrain:
            return y_, argmax_y_, loss.data[0]
        else:
            return y_, argmax_y_

    def predict(self, df):
        batch_size = self.config.batch_size
        self.model.eval()
        if USE_BOW != True:
            sample_encode = Sample_Encode(self.voc)
            # Warning!!
            # sample_encode generator generates wrong batch['y'] when it receives test_df
            g = sample_encode.generator(df, batch_size, shuffle=False, training=False)
        else:
            g = self.bow.batch_generator(df, batch_size, shuffle=False, training=False)
        
        N = df.shape[0] // batch_size + int(df.shape[0] % batch_size != 0)

        y_ = []
        argmax_y_ = []
        try:
            for batch in tqdm(g, total=N):
                if USE_CUDA:
                    batch['X'] = batch['X'].cuda()
                    
                temp_y_, temp_argmax_y_ = self.predict_on_batch(batch)
                y_.append(temp_y_.data)
                argmax_y_.append(temp_argmax_y_)
        except RuntimeError:
            pdb.set_trace()
        # y_ is (N, 2)
        y_ = np.concatenate(y_, axis=0)
        argmax_y_ = np.concatenate(argmax_y_, axis=0)
        self.model.train()
        return y_, argmax_y_

    def split(self, train_df):
        # split 8:2
        N = train_df.shape[0]
        point = N * 8 // 10
        indices = np.random.permutation(N).tolist()
        #
        #pdb.set_trace()
        new_train_df = train_df.iloc[indices[:point]]
        valid_df = train_df.iloc[indices[point:]]
        # dump to pickle
        with open("./data/train_df", "wb") as f1, open("./data/valid_df", "wb") as f2:
            pickle.dump(new_train_df, f1)
            pickle.dump(valid_df, f2) 

        return new_train_df, valid_df
    def load_presplit(self):
        # split training data
        try:
            with open("./data/train_df", "rb") as f1, open("./data/valid_df", "rb") as f2:
                train_df = pickle.load(f1)
                valid_df = pickle.load(f2)
        except FileNotFoundError:
            print("Creating new training pickle .....")
            train_df, valid_df = self.split(train_df)
        return train_df, valid_df

class DNN(nn.Module):
    def __init__(self, vector_dim):
        super(DNN, self).__init__()
        self.linear = nn.Sequential(
                nn.Linear(vector_dim, 500),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Linear(500, 500),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Linear(500, 2)
            )
    def forward(self, X, input_lengths):
        
        result = self.linear(X)
        return result

class DeepJimNetwork(nn.Module):
    def __init__(self, voc_size, embedding_dim, gru_hidden_size, gru_num_layers, bidirectional, \
        WordVectors=None):
        super(DeepJimNetwork, self).__init__()
        ##Tuning hidden size####
        gru_hidden_size = 512
        bidirectional = False
        ########################
        #self.W = nn.Parameter(torch.randn(gru_hidden_size * gru_num_layers * (int(bidirectional)+1), \
        #    gru_hidden_size * (int(bidirectional)+1)), requires_grad=True)
        self.embedding = nn.Embedding(voc_size, embedding_dim)
        #################################################
        self.embedding.weight.requires_grad = False
        #################################################
        if WordVectors is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(WordVectors))
            #self.embedding.weight.requires_grad = False
        self.gru = nn.GRU(embedding_dim, gru_hidden_size, gru_num_layers, dropout=0.4, \
            bidirectional=bidirectional)
        if False:
            dimension = gru_hidden_size * (int(bidirectional)+1) + gru_hidden_size * gru_num_layers * (int(bidirectional)+1)
        else:
            dimension = gru_hidden_size * gru_num_layers * (int(bidirectional)+1)
        '''
        self.linear = nn.Sequential(
            nn.Linear(dimension, 400),
            nn.Dropout(p=0.5),

            nn.Linear(400, 2))
        '''
        self.linear = nn.Sequential(
            nn.Linear(dimension, 32),
            nn.Dropout(p=0.5),

            nn.Linear(32, 2))
    def forward(self, input_seq, input_lengths):
        '''
        input_seq is (T, B)
        input_lengths is (B)
        '''
        # sort input_seq
        #pdb.set_trace()
        # using numpy sort 
        # first sort input_lengths to decreasing order
        
        indices = np.argsort(input_lengths, axis=0)[::-1]      
        input_lengths  = np.take(input_lengths, indices, axis=0).tolist()
        restore_indices = np.argsort(indices, axis=0) # sort to increasing order

        indices = Variable(torch.from_numpy(indices.copy()), requires_grad=False)
        restore_indices = Variable(torch.from_numpy(restore_indices), requires_grad=False)        
        if USE_CUDA:
            indices = indices.cuda()
            restore_indices = restore_indices.cuda()
        # input_seq becomes (B, T)
        input_seq = torch.index_select(input_seq.t(), dim=0, index=indices)
        embedded = self.embedding(input_seq)
        # (B,T) -> (T,B)
        packed = torch.nn.utils.rnn.pack_padded_sequence(torch.transpose(embedded, 0, 1), \
            input_lengths)
        outputs, hidden = self.gru(packed, None)
        hidden = torch.transpose(hidden, 0, 1) # turn it into (batch_size, num_layer * dir, hidden_size)
        hidden = hidden.contiguous()
        hidden = hidden.view(hidden.size()[0], -1) # turn it into (batch_size, num_layer * dir * hidden_size)
        ######
        ## TODO: Attention
        '''
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        # outputs is (B, T, H)
        outputs = torch.transpose(outputs, 0, 1)
        # (B, 2*H) -> (B, H)
        if False:
            transformed_hidden = torch.mm(hidden, self.W)
        else:
            transformed_hidden = hidden
        # Want (B, T, H) multiply (B, H, 1) -> (B, T, 1)
        attention = torch.bmm(outputs, torch.unsqueeze(transformed_hidden, dim=-1))
        # (B, T, 1) -> (B, T)
        attention = torch.squeeze(attention, dim=-1)
        # Mask padded part (B, T)
        mask = torch.ne(input_seq, PAD_token).float()
        # (B, T)
        attention = F.softmax(attention) * mask # WARNING: There will be a lot of zero(padding issue)
        # broadcasting element-wise product (B, T, H) * (B, T, 1) = (B, T, H)
        outputs = outputs * torch.unsqueeze(attention, dim=-1)
        # reduce sum (B, T, H) -> (B, H)
        outputs = torch.sum(outputs, dim=1)
        # concat outputs (B, H) with hidden (B, 2H) -> (B, 3H)
        hidden = torch.cat((outputs, hidden) , dim=1)
        '''
        ######
        
        hidden = self.linear(hidden)
        # restore
        hidden = torch.index_select(hidden, dim=0, index=restore_indices)
        return hidden


