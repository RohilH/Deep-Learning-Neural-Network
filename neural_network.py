import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

"""
    Minigratch Gradient Descent Function to train model
    1. Format the data
    2. call four_nn function to obtain losses
    3. Return all the weights/biases and a list of losses at each epoch
    Args:
        epoch (int) - number of iterations to run through neural net
        w1, w2, w3, w4, b1, b2, b3, b4 (numpy arrays) - starting weights
        x_train (np array) - (n,d) numpy array where d=number of features
        y_train (np array) - (n,) all the labels corresponding to x_train
        num_classes (int) - number of classes (range of y_train)
        shuffle (bool) - shuffle data at each epoch if True. Turn this off for testing.
    Returns:
        w1, w2, w3, w4, b1, b2, b3, b4 (numpy arrays) - resulting weights
        losses (list of ints) - each index should correspond to epoch number
            Note that len(losses) == epoch
    Hints:
        Should work for any number of features and classes
        Good idea to print the epoch number at each iteration for sanity checks!
        (Stdout print will not affect autograder as long as runtime is within limits)
"""
def minibatch_gd(epoch, w1, w2, w3, w4, b1, b2, b3, b4, x_train, y_train, num_classes, shuffle=True):
    print("Epoch: ", epoch)
    print("X-train: ", len(x_train))
    loss = 0
    losses = [0 for i in range(epoch)]
    for iter in range(epoch):
        print("Epoch: ", iter)
        if shuffle == True:
            state = np.random.get_state()
            np.random.shuffle(x_train)
            np.random.set_state(state)
            np.random.shuffle(y_train)
            # indexset = [i for i in range(200)]
            # x_set = [x_train[j + iter*200] for j in indexset]
            # y_set = [y_train[j + iter*200] for j in indexset]

        for i in range(1,int(len(x_train)/200)):
            x_set = x_train[((i - 1) * 200): (i*200)]
            y_set = y_train[((i - 1) * 200): (i*200)]

            # print("Got here")
            loss, w1, w2, w3, w4 = four_nn(x_set, w1, w2, w3, w4, b1, b2, b3, b4, y_set, not shuffle)
        losses[iter] = loss
    print("Losses: ", losses)
    # plt.plot([i for i in range(1,epoch + 1)], losses)
    # plt.show()

    return w1, w2, w3, w4, b1, b2, b3, b4, losses

"""
    Use the trained weights & biases to see how well the nn performs
        on the test data
    Args:
        All the weights/biases from minibatch_gd()
        x_test (np array) - (n', d) numpy array
        y_test (np array) - (n',) all the labels corresponding to x_test
        num_classes (int) - number of classes (range of y_test)
    Returns:
        avg_class_rate (float) - average classification rate
        class_rate_per_class (list of floats) - Classification Rate per class
            (index corresponding to class number)
    Hints:
        Good place to show your confusion matrix as well.
        The confusion matrix won't be autograded but necessary in report.
"""
def test_nn(w1, w2, w3, w4, b1, b2, b3, b4, x_test, y_test, num_classes):
    classifications = four_nn(x_test, w1, w2, w3, w4, b1, b2, b3, b4, y_test, True)
    # print("Classifications: ", classifications)
    # print("Y-Test: ", y_test)
    class_names = np.array(["T-shirt/top","Trouser","Pullover","Dress",
    "Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"])

    plot_confusion_matrix(y_test, classifications, classes=class_names, normalize=True,
                  title='Confusion matrix, with normalization')
    plt.show()
    avg_class_rate = 0.0
    class_rate_per_class = [0.0] * num_classes
    classNumsPerClass = [0] * num_classes
    for i in range(len(y_test)):
        classNumsPerClass[y_test[i]] += 1
        if classifications[i] == y_test[i]:
            avg_class_rate += 1
            class_rate_per_class[y_test[i]] += 1
    avg_class_rate = avg_class_rate / len(y_test)
    for i in range(num_classes):
        class_rate_per_class[i] = class_rate_per_class[i]/classNumsPerClass[i]
    return avg_class_rate, class_rate_per_class

"""
    4 Layer Neural Network
    Helper function for minibatch_gd
    Up to you on how to implement this, won't be unit tested
    Should call helper functions below
"""
def four_nn(x_test, w1, w2, w3, w4, b1, b2, b3, b4, y_test, testBool):
    z1, acache1 = affine_forward(x_test, w1, b1)
    a1, rcache1 = relu_forward(z1)
    z2, acache2 = affine_forward(a1, w2, b2)
    a2, rcache2 = relu_forward(z2)
    z3, acache3 = affine_forward(a2, w3, b3)
    a3, rcache3 = relu_forward(z3)
    F, acache4 = affine_forward(a3, w4, b4)
    if testBool == True:
        classifications = [np.argmax(x) for x in F]
        return classifications
    loss, dF = cross_entropy(F, y_test)
    dA3, dW4, db4 = affine_backward(dF, acache4)
    dZ3 = relu_backward(dA3, rcache3)
    dA2, dW3, db3 = affine_backward(dZ3, acache3)
    dZ2 = relu_backward(dA2, rcache2)
    dA1, dW2, db2 = affine_backward(dZ2, acache2)
    dZ1 = relu_backward(dA1, rcache1)
    dX, dW1, db1 = affine_backward(dZ1, acache1)
    learnRate = 0.1
    w1 = w1 - learnRate*dW1
    w2 = w2 - learnRate*dW2
    w3 = w3 - learnRate*dW3
    w4 = w4 - learnRate*dW4
    return loss, w1, w2, w3, w4

"""
    Next five functions will be used in four_nn() as helper functions.
    All these functions will be autograded, and a unit test script is provided as unit_test.py.
    The cache object format is up to you, we will only autograde the computed matrices.

    Args and Return values are specified in the MP docs
    Hint: Utilize numpy as much as possible for max efficiency.
        This is a great time to review on your linear algebra as well.
"""
def affine_forward(A, W, b):
    # print("A Dim 1: ", len(A), "A Dim 2: ", len(A[0]))
    # print("A Dim 1: ", len(W), "A Dim 2: ", len(W[0]))
    A = np.array(A, dtype=np.float)
    W = np.array(W, dtype=np.float)

    Z = A.dot(W) + b
    cache = (A, W.transpose(), b)
    # print(Z)
    return Z, cache

def affine_backward(dZ, cache):
    dA = dZ.dot(cache[1])
    dW = (cache[0].transpose()).dot(dZ)
    dB = [0 for i in range(len(dZ[0]))]
    dB = np.sum(dZ, axis=0)

    return dA, dW, dB

def relu_forward(Z):
    cache = Z
    Z[Z < 0] = 0
    A = Z
    # print("A: ")
    # print(A)
    return A, cache

def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy = True)
    dZ[Z <= 0 ] = 0

    return dZ

def cross_entropy(F, y):
    sumF = 0
    for i in range(len(F)):
        sumLogF = 0
        sumF += F[i][int(y[i])]
        for j in range(len(F[0])):
            sumLogF += np.exp(F[i][j])
        sumF -= np.log(sumLogF)
    loss = -(sumF)/len(y)
    dF = F
    for i in range(len(F)):
        sumLogF = 0
        for j in range(len(F[0])):
            sumLogF += np.exp(F[i][j])
        for j in range(len(F[0])):
            dF[i][j] = -(1*(j == y[i]) - np.exp(F[i][j])/sumLogF)/len(y)
    return loss, dF
