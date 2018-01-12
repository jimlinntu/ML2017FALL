from import_modules import *
def load_image_npy(path):
    image_array = np.load(path)
    return image_array

def load_test_case_csv(path):
    test_case_df = pd.read_csv(path)
    return test_case_df

# print to csv
def print_to_csv(y_, filename):
    data = [[index, issame] for index, issame in enumerate(y_)]
    df = pd.DataFrame(data=data, columns=["ID", "Ans"])
    df.to_csv(filename, index=False)