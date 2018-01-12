import numpy as np
from skimage import io
import pdb
import argparse
import os
def main():
    parser = argparse.ArgumentParser(description='ML2017/hw6')
    parser.add_argument('img_folder', type=str, help='img/')
    parser.add_argument('reconstruct_img', type=str, help="_.jpg")
    args = parser.parse_args()
    # 
    image_array = []
    print("Reading 415 images" + "."*70)
    for image_paths in os.listdir(args.img_folder):
        try:
            image = io.imread(os.path.join(args.img_folder, image_paths))
            image_array.append(image)
        except:
            continue
    image_array = np.stack(image_array).astype(float)
    # mean
    mean = np.mean(image_array , axis=0)
    # flatten
    mean = mean.reshape(-1)
    # fi array
    A = np.zeros((600 * 600 * 3, 415))
    for index, image in enumerate(image_array):
        A[:, index] = image.reshape(-1) - mean
    
    #
    top_k = 4  # Use first 4 eigen vector to reconstruct
    if False:        
        # SVD decomposition
        print("SVD Decomposition" + "."*70)
        U, s, V = np.linalg.svd(A, full_matrices=False)
        first_k_U = U[:, :top_k].copy()
    else:
        print("Eigen Decomposition" + "."*70)
        
        L = A.T.dot(A)
        w, v = np.linalg.eig(L)
        # sort eigenvalue
        arg_index = np.argsort(np.absolute(w))
        # biggest four index
        top_k_index = arg_index[-top_k:]
        # compute eigen vectors
        first_k_U = A.dot(v[:, top_k_index])
        # normalize
        length = np.sqrt(np.sum(first_k_U ** 2, axis=0))
        first_k_U = first_k_U / length
    # 
    index = int(args.reconstruct_img.split(".")[0])
    print("Reconstruct {} image".format(index))
    
    y_ = 0
    reconstruct_img = io.imread(os.path.join(args.img_folder, args.reconstruct_img))
    reconstruct_img = reconstruct_img.reshape(-1) - mean
    coefficient = reconstruct_img.dot(first_k_U)
    print("coefficient {}".format(coefficient))
    for i in range(top_k):
        y_ += coefficient[i] * first_k_U[:, i]
    y_ += mean
    y_ = y_.reshape(600, 600, 3)
    y_ -= np.min(y_)
    y_ /= np.max(y_)
    y_ = (y_ * 255).astype(np.uint8)
    # save to image
    io.imsave("reconstruction.jpg", y_, quality=100)

if __name__ == '__main__':
    main()