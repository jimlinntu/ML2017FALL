from model import *
from util import *
from import_modules import *

def main():
    image_array = load_image_npy("../data/image_cluster/image.npy")
    test_case_df = load_test_case_csv("../data/image_cluster/test_case.csv")
    visualize_index = [0, 500, 4829, 7764]
    now = dt.datetime.now()
    timeline = "time: %d%02d%02d_%02d%02d" % (now.year, now.month, now.day, now.hour, now.minute)
    model = AutoEncoder_DNN2()
    optimizer = optim.Adam(filter(lambda p:p.requires_grad, model.parameters()), 
            lr=0.001)
    criterion = nn.MSELoss()
    model.cuda()
    batch_size = 200
    min_loss = float("inf")
    mean = np.mean(image_array, axis=0)
    std = np.std(image_array, axis=0, ddof=1)
    #pdb.set_trace()
    image_array = (image_array - mean) / (std + 1e-6)
    try:
        # epoch
        for i in range(10000):
            print("Epoch {} out of 1000".format(i+1))
            g = batch_generator(image_array, batch_size=batch_size)
            total_loss = 0.
            quotient = image_array.shape[0] // batch_size
            for batch in g:
                optimizer.zero_grad()
                X_, code = model.forward(batch["X"])
                loss = criterion.forward(X_, batch["X"])
                loss.backward()
                optimizer.step()
                total_loss += loss.data[0]
            if total_loss < min_loss:
                min_loss = total_loss
                torch.save(model.state_dict(), "./param/best"+str(timeline))
                '''
                for index, index_index in enumerate(visualize_index):
                    plt.imsave("result{}.jpg".format(index), (image_array[index_index] * std + mean).reshape(28, 28), cmap="Greys")
                    reconstruct = model(Variable(torch.from_numpy(image_array[index_index]).float()).cuda())[0]
                    reconstruct = (reconstruct.cpu().data.numpy()* std + mean).reshape(28, 28)
                    plt.imsave("result_{}.jpg".format(index), reconstruct, cmap="Greys")
                '''
            print("Last Loss {}".format(total_loss/quotient))
            print("Min Loss {}".format(min_loss/quotient))
    except KeyboardInterrupt:
        pass
    # save
    torch.save(model.state_dict(), "./param/"+str(timeline))




def batch_generator(X, batch_size, shuffle=True, volatile=False):
    N = X.shape[0]
    if shuffle == True:
        random_permutation = np.random.permutation(N)
        new_X = X[random_permutation]
    else:
        new_X = X.copy()
    quotient = N // batch_size
    remainder = N % batch_size
    for i in range(quotient+int(remainder!=0)):
        batch = {}
        batch["X"] = Variable(torch.from_numpy(new_X[i*batch_size:(i+1)*batch_size]).float(),
            volatile=volatile)
        if torch.cuda.is_available():
            batch["X"] = batch["X"].cuda()
        yield batch
if __name__ == '__main__':
    main()