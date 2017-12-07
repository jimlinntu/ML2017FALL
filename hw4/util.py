from import_modules import *
class Preprocess:
    '''
        Preprocess raw data
    '''
    def __init__(self):
        self.regex_remove_punc = re.compile('[%s]' % re.escape(string.punctuation))
        pass
    def unicodeToAscii(self, s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )

    def normalizeString(self, sentence):
        sentence = self.unicodeToAscii(sentence.strip())
        #sentence = self.unicodeToAscii(sentence.lower().strip())
        # remove punctuation
        if False:
            sentence = self.regex_remove_punc.sub('', sentence)
        sentence = re.sub(r"([.!?])", r" \1", sentence)
        sentence = re.sub(r"[^a-zA-Z.!?]+", r" ", sentence)
        sentence = re.sub(r"\s+", r" ", sentence).strip()
        return sentence

    def remove_punctuation(self, sentence):
        sentence = self.regex_remove_punc.sub('', sentence)
        return sentence

    def read_txt(self, train_filename, test_filename, train_filename_no_label):
        train_df = None
        test_df = None
        train_df_no_label = None
        
        if train_filename is not None:
            train_df = pd.read_csv(train_filename, header=None, names=["label", "seq"], sep="\+\+\+\$\+\+\+",
                                  engine="python")
            # remove puncuation
            #train_df["seq"] = train_df["seq"].apply(lambda seq: self.normalizeString(seq))
            
        
        if test_filename is not None:
            with open(test_filename, "r") as f:
                reader = csv.reader(f, delimiter=",")
                rows = [[row[0], ",".join(row[1:])] for row in reader]
                test_df = pd.DataFrame(rows[1:], columns=rows[0]) # first row is column name
            # remove puncuation
            #test_df["text"] = test_df["text"].apply(lambda seq: self.normalizeString(seq))
        if train_filename_no_label is not None:
            train_df_no_label = pd.read_csv(train_filename_no_label, sep="\n", header=None, names=["seq"])
            train_df_no_label.insert(loc=0, column="nan", value=0)
            # remove puncuation
            #train_df_no_label["seq"] = train_df_no_label["seq"].apply(lambda seq: self.normalizeString(seq))
        
        return train_df, test_df, train_df_no_label

class Sample_Encode:
    '''
        Transform 
    '''
    def __init__(self, voc):
        self.voc = voc

    def sentence_to_index(self, sentence):
        encoded = list(map(lambda token: self.voc.word2index[token] if token in self.voc.word2index \
            else UNK_token, sentence))
        return encoded

    def pad_batch(self, index_batch):
        '''
            Return padded list with size (B, Max_length)
        '''
        return list(itertools.zip_longest(*index_batch, fillvalue=PAD_token))

    def batch_to_Variable(self, sentence_batch, training=True):
        '''
            Input: a numpy of sentence
            ex. ["i am a", "jim l o "]

            Output: a torch Variable and sentence lengths
        '''
        # split sentence
        sentence_batch = sentence_batch.tolist()
        
        # apply
        for training_sample in sentence_batch:
            # split training sentence
            training_sample[1] = training_sample[1].strip(" ").split(" ")

        # encode batch
        index_label_batch = [(training_sample[0], self.sentence_to_index(training_sample[1])) \
            for training_sample in sentence_batch]

        # sort sentence batch (in order to fit torch pack_pad_sequence)
        #index_label_batch.sort(key=lambda x: len(x[1]), reverse=True) 
        
        # index batch
        index_batch = [training_sample[1] for training_sample in index_label_batch]
        label_batch = [training_sample[0] for training_sample in index_label_batch]

        # record batch's length
        lengths = [len(indexes) for indexes in index_batch]

        # padded batch
        padded_batch = self.pad_batch(index_batch)

        # transform to Variable
        if training:
            pad_var = Variable(torch.LongTensor(padded_batch), volatile=False)
        else:
            pad_var = Variable(torch.LongTensor(padded_batch), volatile=True)

        # label
        if training:
            label_var = Variable(torch.LongTensor(label_batch), volatile=False)
        else:
            label_var = None

        
        return pad_var, label_var, lengths
    
    def generator(self, df, batch_size, shuffle=False, training=True):
        '''
        Return sample batch Variable
            batch['X'] is (T, B)
        '''
        df_matrix = df.as_matrix()
        if shuffle == True:
            random_permutation = np.random.permutation(len(df['seq']))
            
            # shuffle
            df_matrix = df_matrix[random_permutation]
        #
        quotient = df.shape[0] // batch_size
        remainder = df.shape[0] - batch_size * quotient

        for i in range(quotient):
            batch = {}
            X, y, lengths = self.batch_to_Variable(df_matrix[i*batch_size:(i+1)*batch_size], training)
            batch['X'] = X
            batch['y'] = y
            batch['lengths'] = lengths
            yield batch
            
        if remainder > 0: 
            batch = {}
            X, y, lengths = self.batch_to_Variable(df_matrix[-remainder:],training)
            batch['X'] = X
            batch['y'] = y
            batch['lengths'] = lengths
            yield batch

class BOW():
    def __init__(self):
        self.vectorizer = CountVectorizer(max_features=10000)

    def unicodeToAscii(self, s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )
    def remove_punctuation(self, sentence):
        sentence = self.unicodeToAscii(sentence)
        sentence = re.sub(r"([.!?])", r" \1", sentence)
        sentence = re.sub(r"[^a-zA-Z.!?]+", r" ", sentence)
        sentence = re.sub(r"\s+", r" ", sentence).strip()
        return sentence

    def fit(self, train_df, train_df_no_label):
        # prepare copus
    
        corpus = list(map(lambda x: self.remove_punctuation(x), train_df['seq']))
        corpus += list(map(lambda x: self.remove_punctuation(x), train_df_no_label['seq']))
        print("BOW fitting")
        self.vectorizer.fit(corpus)
        self.dim = len(self.vectorizer.get_feature_names())
        print("BOW fitting done")
        return self

    def batch_generator(self, df, batch_size, shuffle=True, training=True):
         # (B, Dimension)
        N = df.shape[0]
        df_matrix = df.as_matrix()
        
        if shuffle == True:
            random_permutation = np.random.permutation(N)
            
            # shuffle
            X = df_matrix[random_permutation, 1]
            y = df_matrix[random_permutation, 0].astype(int) # 0 is label's index
        else:
            X = df_matrix[:, 1]
            y = df_matrix[:, 0].astype(int)
        #
        quotient = X.shape[0] // batch_size
        remainder = X.shape[0] - batch_size * quotient

        for i in range(quotient):
            batch = {}
            batch_X = self.vectorizer.transform(X[i*batch_size:(i+1)*batch_size]).toarray()
            batch['X'] = Variable(torch.from_numpy(batch_X)).float()
            if training:
                batch_y = y[i*batch_size:(i+1)*batch_size]
                batch['y'] = Variable(torch.from_numpy(batch_y))
            else:
                batch['y'] = None
            batch['lengths'] = None
            yield batch
            
        if remainder > 0: 
            batch = {}
            batch_X = self.vectorizer.transform(X[-remainder:]).toarray()
            batch['X'] = Variable(torch.from_numpy(batch_X)).float()
            if training:
                batch_y = y[-remainder:]
                batch['y'] = Variable(torch.from_numpy(batch_y))
            else:
                batch['y'] = None
            batch['lengths'] = None
            yield batch

class Voc:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = ["PAD", "UNK"] # might be changed
        self.n_words = 10000 + 2 # might be changed

    def unicodeToAscii(self, s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )
    def remove_punctuation(self, sentence):
        sentence = self.unicodeToAscii(sentence)
        sentence = re.sub(r"([.!?])", r" \1", sentence)
        sentence = re.sub(r"[^a-zA-Z.!?]+", r" ", sentence)
        sentence = re.sub(r"\s+", r" ", sentence).strip()
        return sentence

    def fit(self, train_df, train_df_no_label, USE_Word2Vec=True):
        print("Voc fitting...")
        
        # tokenize
        tokens = []
        sentences = []
        
        for sequence in train_df["seq"]:
            token = sequence.strip(" ").split(" ")
            tokens += token
            sentences.append(token)


        for sequence in train_df_no_label["seq"]:
            token = sequence.strip(" ").split(" ")
            tokens += token
            sentences.append(token)

        # Using Word2Vec
        if USE_Word2Vec:
            dim = 100
            print("Word2Vec fitting")
            model = Word2Vec(sentences, size=dim, window=5, min_count=20, workers=20, iter=20)
            print("Word2Vec fitting finished....")
            # gensim index2word 
            self.index2word += model.wv.index2word
            self.n_words = len(self.index2word)
            # build up numpy embedding matrix
            embedding_matrix = [None] * len(self.index2word) # init to vocab length
            embedding_matrix[0] = np.random.normal(size=(dim,))
            embedding_matrix[1] = np.random.normal(size=(dim,))
            # plug in embedding
            for i in range(2, len(self.index2word)):
                embedding_matrix[i] = model.wv[self.index2word[i]]
                self.word2index[self.index2word[i]] = i
            
            # 
            self.embedding_matrix = np.array(embedding_matrix)
            return
        else:
            # Counter
            counter = Counter(tokens)
            voc_list = counter.most_common(10000)

            for i, (voc, freq) in enumerate(voc_list):
                self.word2index[voc] = i+2
                self.index2word[i+2] = voc
                self.word2count[voc] = freq

def print_to_csv(y_, filename):
    d = {"id":[i for i in range(len(y_))],"label":list(map(lambda x: str(x), y_))}
    df = pd.DataFrame(data=d)
    df.to_csv(filename, index=False)

