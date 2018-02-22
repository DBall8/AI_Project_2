import numpy as np
from sklearn import tree
import graphviz as gp
from PIL import Image


# Converts a one-hots vector for a image label back into its label
# INPUT - v - the one-hot vector
# OUTPUT - the number the label represented
def getLabel(v):
    for i in range(10):
        if(v[i] == 1):
            return i

# Calculates precision and recall for each label
# INPUT - conf - the confusion matrix
# OUTPUT - precision - a vector of precision scores where its index
#       corresponds to its label (i.e. index 0 is 0's precision score)
# OUTPUT - recall - a vector of recall scores where its index
#       corresponds to its label (i.e. index 0 is 0's recall score)
def getPandR(conf):
    # Initiate vectors
    recall = np.zeros((10))
    precision = np.zeros((10))

    # Calculate for each label
    for c in range(10):
        TP = conf[c,c] # True positives
        FN = 0 # sum of false negatives
        FP = 0 # sum of false positives

        # sum the false negatives for the current label
        for i in range(10):
            if i != c:
                FN += conf[c,i]
        # sum the false positives for each label
        for i in range(10):
            if i != c:
                FP += conf[i,c]
        # calculate precision and recall for each label
        recall[c] = TP / (TP + FN)
        precision[c] = TP / (TP + FP)
    return precision, recall

# Get the overall accuracy of a Decision Tree
# INPUT - conf - the confusion matrix of the tree
# OUTPUT - the accuracy score of the decision tree
def getAccuracy(conf):
    success = 0
    total = 0
    # sum the correct and total predictions
    for i in range(10):
        for j in range(10):
            total += conf[i,j]
            if(i == j):
                success += conf[i,i]
    # calculate accuracy
    return success/total

# Compare two vectors, returns a vector of their differences
def compare(v1, v2):
    v3 = np.zeros((10))
    for i in range(10):
        v3[i] = v1[i] - v2[i]
    return v3

# Builds a decision tree from the traindata and trainlabels sets
# INPUT - testdata - the data to use to test the decision tree
# INPUT - testlabels - the labels for the testdata set
# INPUT (optional) - depth - the maximum depth of the decision tree
# INPUT (optional) - split - the minimimum number of samples in order to create
#       a new split in the tree
# INPUT (optional) - leaf - the minimimum number of samples at a leaf node
# OUTPUT - conf - the confusion matrix resultant from running the testdata through
#       the decision tree
def makeTree(testdata, testlabels, depth=-1, split=2, leaf=1):

    # Use the training data set
    global traindata, trainlabels

    # Make the decision tree
    # Use the depth option if a depth was given
    if(depth > 0):
        dt = tree.DecisionTreeClassifier(max_depth=depth, min_samples_split=split, min_samples_leaf=leaf)
    else:
        dt = tree.DecisionTreeClassifier(min_samples_split=split, min_samples_leaf=leaf)

    # Fit the tree to the training data set
    dt = dt.fit(traindata, trainlabels)

    # Get the predicted labels from running the test data through the decision tree
    predictions = dt.predict(testdata)

    # Build the confusion matrix, also saving 1 mispredicted image for each label
    saved = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] # entry becomes 1 for the label with a image already saved
    # Initiate confusion matrix
    conf = np.zeros((10,10))
    # For each label
    for i in range(len(testlabels)):
        # Get the real and guessed labels
        reallabel = getLabel(testlabels[i])
        predictlabel = getLabel(predictions[i])
        # Save a mispredicted image if labels dont match
        if(saved[reallabel] == 0 and reallabel != predictlabel):
            saveIm(testdata[i], "Real" + str(reallabel) + "Predict" + str(predictlabel) + ".png")
            saved[reallabel] = 1

        # Increment the corresponding entry in the confusion matrix
        conf[reallabel, predictlabel] += 1
    return conf


# Save a confusion matrix and its corresponding precision and recall scores to
# .csv files
# INPUT - conf - confusion matrix to save
def saveData(conf):

    # Save confusion matrix
    file = open('confusion_matrix.csv', "w")
    for r in conf:
        for i in r:
            file.write(str(i) + ',')
        file.write('\n')
    file.close()

    # Get precision and recall
    p, r = getPandR(conf)

    # Save precisions
    file = open('precision.csv', "w")
    for i in p:
        file.write(str(i) + ',')
    file.close()

    # Save recalls
    file = open('recall.csv', "w")
    for i in r:
        file.write(str(i) + ',')
    file.close()

# Save an image from its pixel array
# INPUT - im - image array (1 by 784) to save
# INPUT - filename - name of file to save
def saveIm(im, filename):
    # reshape
    im = np.reshape(im, (28,28))
    #save
    Image.fromarray(im).save(filename, mode="L")


# load images from file
images = np.load('images.npy')
d = np.shape(images)
numImages = d[0]

# Create additional features for each image
features = []
for im in images:
    f = [] # feature vector for each image

    f.append(sum(im))

    # add the sum of each row as a geature
    for r in im:
        f.append(sum(r))
    # add the sum of each column as a feature
    for i in range(28):
        s = 0
        for r in im:
            s += r[i]
        f.append(s)

    # add the sum of each 7 by 7 block as a feature
    for i in range(0, 28, 7):
        for j in range(0, 28, 7):
            s = 0
            for n in range(i, i+7):
                for m in range(j, j+7):
                    s += im[n,m]
            f.append(s)
    features.append(f)

# Flatten each image array
images = np.reshape(images, (numImages, 784))

# load corresponding labels
labels = np.load('labels.npy')

# Counts holding the amount of data saved in each set
trainCount = 0;
validCount = 0
testCount = 0;

# Array that sorts each data entry by the label
# The index corresponds to the label i.e. index 3 has the data of label 3
classedImages = [[],[],[],[],[],[],[],[],[],[]]

# Sort each image by class, while also adding the features vector to the end of it
for i in range(numImages):
    im = images[i]
    np.append(im, features[i])
    classedImages[labels[i]].append(im)

# Arrays for each subset of data
traindata = []
validdata = []
testdata = []

# Matching arrays for the labels of the above
trainlabels = []
validlabels = []
testlabels = []

# Fill each subset of the main dataset with the appropriate number of images
for c in range(10):
    # Get the number of images of this label
    numThisClass = len(classedImages[c])
    # reset counts
    trainCount = 0;
    validCount = 0
    testCount = 0;
    # insert each datapoint of this class into one of the three datasets
    for i in range(numThisClass):
        # Insert into the training set if training set isnt full
        if(trainCount < 0.6*numThisClass):
            traindata.append(classedImages[c][i])
            # convert the label into the one-hots vector
            v = [0]*10
            v[c] = 1
            trainlabels.append(v)
            trainCount += 1
        # Insert into the validation set if validation set isnt full
        elif(validCount < 0.15*numThisClass):
            validdata.append(classedImages[c][i])
            # convert the label into the one-hots vector
            v = [0]*10
            v[c] = 1
            validlabels.append(v)
            validCount += 1
        # Insert into the testing set if testing set isnt full
        else:
            testdata.append(classedImages[c][i])
            # convert the label into the one-hots vector
            v = [0]*10
            v[c] = 1
            testlabels.append(v)
            testCount += 1
