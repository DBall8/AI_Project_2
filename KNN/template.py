import numpy as np
import numpy.random
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors


def printIMG(img):
    image = img.reshape([28,28])
    plt.gray()
    plt.imshow(image)
    plt.show()

def processIMG(imagePath, labelsPath):
    images = np.load(imagePath)
    image_vectors = []

    image_vectors = images.reshape(6500, 784)
    image_vectors = np.array(image_vectors)

    labels = np.load(labelsPath)
    labels_flat = labels.reshape(6500, 1)
    label_vectors = []
    one_hot_labels = keras.utils.to_categorical(labels_flat, num_classes=10)
    label_vectors = np.array(one_hot_labels)

    data = np.column_stack((label_vectors, image_vectors))
    np.random.shuffle(data)

    one_hot_labels = data[:,0:10]
    flattend_images = data[:,10:]

    data_size = 6500
    training_size = int(data_size * 0.60)
    print ("Training Set size = ", training_size)
    val_size = int(data_size * (0.65 + 0.15))
    print ("Validation set index = ", val_size)

    num_classes = np.unique(labels).shape[0]
    x_train, x_val, x_test = flattend_images[:training_size], flattend_images[training_size:val_size], flattend_images[val_size:]
    y_train, y_val, y_test = one_hot_labels[:training_size], one_hot_labels[training_size: val_size], one_hot_labels[val_size:]
    return x_train, y_train, x_val, y_val, x_test, y_test


# Model Template

x_train, y_train, x_val, y_val, x_test, y_test = processIMG("images.npy","labels.npy")

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(x_train, y_train)

expected = y_test

predicted = knn.predict(x_test)

a = knn.kneighbors(n_neighbors=3, return_distance=False)
printIMG(x_train[a[0][0]])
printIMG(x_train[a[0][1]])
printIMG(x_train[a[0][2]])

print("n=3")
print("Classification")
print(metrics.classification_report(expected, predicted))
print ("Confusion Matrix")
y_act = list(map(np.argmax, expected))
y_pred = list(map(np.argmax, predicted))
print(confusion_matrix(y_act,y_pred, [0,1,2,3,4,5,6,7,8,9]))



