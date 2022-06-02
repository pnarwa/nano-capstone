####################################################################################
##This file contains definitions for some common functions used across the notebooks
#####################################################################################
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend


def create_model(input_shape, output_shape):

    '''
    INPUT:
    input_shape (tuple) : tuple specifying the shape of the input image
    output_shape (int) : integer specifying the number of class labels

    OUTPUT:
    tensorflow.python.keras.engine.sequential.Sequential : the Sequential model

    Description:
    The function is used to build a CNN model using the Keras Sequential API.
    The model architecture consists of 3 blocks, each block made of 2 CNN layers
    followed by a MaxPool layer, and a Dropout layer.
    The output from the 3rd block is flatenned and fed to two fully connected (FC) layers.
    The last layer is the output layer performing the final classification of the images.
    '''

    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu',  padding='same', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu',  padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu',  padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(output_shape, activation='sigmoid'))

    #compile model
    opt = SGD(learning_rate=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=[fbeta_score])
    return model


### We make use of the precision, recall and fbeta_score functions implemented by the Keras team at [Keras GitHub]
###(https://github.com/keras-team/keras/blob/4fa7e5d454dd4f3f33f1d756a2a8659f2e789141/keras/metrics.py#L134)
def precision(y_true, y_pred):
    '''
    INPUT:
    y_true (numpy.array) : numpy array containing the true label values
    y_pred (numpy.array) : numpy array containg the predicted lablel values

    OUTPUT:
    float : the precision score

    Description:
    The function computes the precision score based on the predicted labels
    and true labels passed as input parameters.
    '''

    true_positives = backend.sum(backend.round(backend.clip(y_true * y_pred, 0, 1)))
    predicted_positives = backend.sum(backend.round(backend.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + backend.epsilon())
    return precision


def recall(y_true, y_pred):
    '''
    INPUT:
    y_true (numpy.array) : numpy array containing the true label values
    y_pred (numpy.array) : numpy array containg the predicted lablel values

    OUTPUT:
    float : the recall score

    Description:
    The function computes the recall score based on the predicted labels
    and true labels passed as input parameters.
    '''

    true_positives = backend.sum(backend.round(backend.clip(y_true * y_pred, 0, 1)))
    possible_positives = backend.sum(backend.round(backend.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + backend.epsilon())
    return recall


def fbeta_score(y_true, y_pred, beta=1):
    '''
    INPUT:
    y_true (numpy.array) : numpy array containing the true label values
    y_pred (numpy.array) : numpy array containg the predicted label values
    beta (int) : integer parameter indicating the weight to be assigned to the recall metric

    OUTPUT:
    float : the F-beta score

    Description:
    The function computes the F-beta score based on the predicted labels
    and true labels passed as input parameters. For the default value of beta=1,
    it is the F1-score.
    '''

    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if backend.sum(backend.round(backend.clip(y_true, 0, 1))) == 0:
        return 0.0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + backend.epsilon())
    return fbeta_score
