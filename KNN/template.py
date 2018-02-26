import numpy as np
import numpy.random
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors


def printIMG(image):
    image = image.reshape([28,28])
    plt.gray()
    plt.imshow(image)
    plt.show()

def processIMG(imagePath, labelsPath):
    images = np.load(imagePath)
    image_vectors = []
    image_vectors = images.reshape(6500, 784)
    image_vectors = np.array(image_vectors)

    labels = np.load(labelsPath)
    label = labels.reshape(6500, 1)
    label_vectors = []
    classify_label = keras.utils.to_categorical(label, num_classes=10)
    label_vectors = np.array(classify_label)
    data = np.column_stack((label_vectors, image_vectors))
    np.random.shuffle(data)
    classify_label = data[:,0:10]
    end_images = data[:,10:]

    data_length = 7000
    train = int(data_length * 0.60)
    value = int(data_length * (0.65 + 0.15))

    num_classes = np.unique(labels).shape[0]
    x_train, x_val, x_test = end_images[:train], end_images[train:value], end_images[value:]
    y_train, y_val, y_test = classify_label[:train], classify_label[train: value], classify_label[value:]
    return x_train, y_train, x_val, y_val, x_test, y_test


# Model Template

x_train, y_train, x_val, y_val, x_test, y_test = processIMG("images.npy","labels.npy")

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)
expected = y_test
predicted = knn.predict(x_test)

a = knn.kneighbors(n_neighbors=3, return_distance=False)
#printIMG(x_train[a[0][0]])
#printIMG(x_train[a[0][1]])
#printIMG(x_train[a[0][2]])
#printIMG(x_train[a[0][3]])

print("k=3")
print("Classification Report")
print(metrics.classification_report(expected, predicted))
print ("Confusion Matrix")
y_act = list(map(np.argmax, expected))
y_pred = list(map(np.argmax, predicted))
print(confusion_matrix(y_act,y_pred, [0,1,2,3,4,5,6,7,8,9]))



