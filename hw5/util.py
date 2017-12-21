from import_modules import *
def batch_generator(X, y, batch_size, shuffle=True, training=True):
    is_y = y is not None
    N = X.shape[0]
    if shuffle == True:
        random_permutation = np.random.permutation(N)
        X = X[random_permutation]
        y = y[random_permutation] if is_y else None
    #
    quotient = N // batch_size
    remainder = N % batch_size
    for i in range(quotient):
        batch = {}
        batch["X"] = Variable(torch.from_numpy(X[i*batch_size:(i+1)*batch_size]).long(),
            volatile=not training)
        batch["y"] = Variable(torch.from_numpy(y[i*batch_size:(i+1)*batch_size]).float(),
            volatile=not training) if is_y else None
        if torch.cuda.is_available():
            batch["X"] = batch["X"].cuda()
            batch["y"] = batch["y"].cuda() if is_y else None
        yield batch
    if remainder != 0:
        batch = {}
        batch["X"] = Variable(torch.from_numpy(X[-remainder:]).long(),
            volatile=not training)
        batch["y"] = Variable(torch.from_numpy(y[-remainder:]).float(),
            volatile=not training)if is_y else None
        if torch.cuda.is_available():
            batch["X"] = batch["X"].cuda()
            batch["y"] = batch["y"].cuda()if is_y else None
        yield batch

def movie_genre_one_hot(movie_csv, M, movies2index):
    df = pd.read_csv(movie_csv, header=0, delimiter="::")
    
    genre2index = {}
    genre_count = 0
    for genre_list in df["Genres"]:
        genre_list = genre_list.split("|")
        for genre in genre_list:
            if genre not in genre2index:
                genre2index[genre] = genre_count
                genre_count += 1

    assert len(genre2index) == genre_count
    #
    augment_one_hot_embedding = np.zeros((M, genre_count))
    # one hot encode
    for genre_list, movieID in zip(df["Genres"], df["movieID"]):
        genre_list = genre_list.split("|")
        for genre in genre_list:
            if movieID in movies2index:
                # turn on the bit which it belongs to
                augment_one_hot_embedding[movies2index[movieID], genre2index[genre]] = 1
    pdb.set_trace()
    return augment_one_hot_embedding

def read_train(csv_file, test_csv_file):
    if csv_file is None:
        return None
    df = pd.read_csv(csv_file, header=0)
    test_df = pd.read_csv(test_csv_file, header=0)

    userid = pd.concat([df["UserID"], test_df["UserID"]]).unique()
    movies = pd.concat([df["MovieID"], test_df["MovieID"]]).unique()
    U = len(userid)
    M = len(movies)
    # lookup dictionary
    train = {}
    userid2index = {userid: index for index, userid in enumerate(userid)}
    movies2index = {movieid: index for index, movieid in enumerate(movies)}
    # transform userid to index
    df["UserID"] = df["UserID"].apply(lambda userid: userid2index[userid])
    df["MovieID"] = df["MovieID"].apply(lambda movieid: movies2index[movieid])
    # X is (N, 2)
    train["X"] = df[["UserID", "MovieID"]].as_matrix()
    # TODO: Normalize rating (Try minmax normalization first)
    train["y"] = np.squeeze(df[["Rating"]].as_matrix(), axis=-1)

    train["U"] = U
    train["M"] = M
    train["userid2index"] = userid2index
    train["movies2index"] = movies2index
    return train

def read_test(csv_file, userid2index, movies2index):
    if csv_file is None:
        return None
    df = pd.read_csv(csv_file, header=0)
    test = {}
    # transform userid to index
    df["UserID"] = df["UserID"].apply(lambda userid: userid2index[userid])
    df["MovieID"] = df["MovieID"].apply(lambda movieid: movies2index[movieid])
    test["X"] = df[["UserID", "MovieID"]].as_matrix()
    return test
def print_to_csv(y_, filename):
    array = [[index+1, rating] for index, rating in 
            zip(range(len(y_)), map(lambda x: str(x), y_))]
    array = np.array(array)
    df = pd.DataFrame(data=array, columns=["TestDataID", "Rating"])
    df.to_csv(filename, index=False)
