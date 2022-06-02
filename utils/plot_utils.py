####################################################################################
##This file contains definitions for some common functions used across the notebooks
#####################################################################################
import matplotlib.pyplot as plt
from matplotlib.image import imread
import numpy as np

def plot_model_perf(history):
    '''
    INPUT:
    history (tf.keras.callbacks.History) :
           the history callback object returned by the model.fit() method

    OUTPUT:
    tf.keras.callbacks.History : a history callback object which keeps track of loss,
                                accuracy, other training metrics for each epoch

    Description:
    The history callback object returned by the model.fit() method keeps track of
    loss, performance and other training metrics.
    The function uses the history callback and the matplotlib library to plot the
    loss and the F1-score at the end of each training ecpoch for both the training
    dataset and the validation dataset.
    '''

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(14,6))
    ax1.plot(history.history['loss'], color='blue', label='Train');
    ax1.plot(history.history['val_loss'], color='red', label='Test')
    ax1.set_xlabel('Epoch');
    ax1.set_ylabel('Loss');
    ax1.set_title('Cross Entropy Loss');
    ax1.legend()

    ax2.plot(history.history['fbeta_score'], color='blue', label='Train');
    ax2.plot(history.history['val_fbeta_score'], color='red', label='Test')
    ax2.set_xlabel('Epoch');
    ax2.set_ylabel('F1-score');
    ax2.set_title('F1-Score');
    ax2.legend()
    plt.show()


def plot_predictions(X_test, Y_test, Y_pred, categories, num):
    '''
    INPUT:
    X_test (numpy.array) : numpy array containing the image data for the test dataset
    Y_test (numpy.array) : numpy array containing the true labels for the test dataset
    Y_pred (numpy.array) : numpy array containing the predicted labels for the test dataset
    categories (list) : list containing the class labels

    OUTPUT:

    Description:
    The function plots the images and predicted labels for a subset of the test dataset
    and compares with the true labels. It uses the matplot library to plot the images.
    '''

    fig = plt.figure(figsize=(16,12))
    plt.subplots_adjust(hspace=0.4)
    dlim = ','
    for i in range(num):
        ax = fig.add_subplot(3, 4, i + 1, xticks=[], yticks=[])
        ax.imshow(np.squeeze(X_test[i]))
        pred_ids = np.where(Y_pred[i] == 1)
        pred_tags = dlim.join([categories[idx] for idx in pred_ids[0]])
        true_ids = np.where(Y_test[i] == 1)
        true_tags = dlim.join([categories[idx] for idx in true_ids[0]])
        if np.array_equal(pred_ids, true_ids) == False:
            ax.set_title(pred_tags, color='red')
        else:
            ax.set_title(pred_tags, color='blue')
        ax.set_xlabel(true_tags, color='blue');
