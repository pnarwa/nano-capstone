####################################################################################
##This file contains definitions for some common functions used across the notebooks
#####################################################################################
import glob
import numpy as np

def load_data(path):
    '''
    INPUT:
    path (str) : the file-system path where the npz files are saved.

    OUTPUT:
    numpy.array : numpy array containing the image data
    numpy.array : numpy array containing the target labels

    Description:
    The function loads the npz files created during the pre-processing stage,
    retreives the data from the files, and stacks the data to create
    numpy arrays corresponding to the images and labels.
    '''

    x_list = []
    y_list = []
    for name in glob.glob(path):
        amz_imgaes = np.load(name)
        x, y = amz_imgaes['X'], amz_imgaes['Y']
        x_list.append(x)
        y_list.append(y)
    x_tup = tuple(x_list)
    y_tup = tuple(y_list)
    X = np.vstack(x_tup)
    Y = np.vstack(y_tup)
    return X, Y


def rescale_data(X_data):
    '''
    INPUT:
    X_data (numpy.array) : numpy array containing the image data 

    OUTPUT:
    numpy.array : numpy array containing the normalized dataset

    Description:
    The function is used to normalize the data so that each pixel
    in the dataset is scaled to a value between [0,1]
    '''

    X_data = X_data.astype('float32')/255
    return X_data

