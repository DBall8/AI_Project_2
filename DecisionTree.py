import numpy as np
from sklearn import tree

images = np.load('images.npy')
d = np.shape(images)
numImages = d[0]
images = np.reshape(images, (numImages, 784))

l = np.load('labels.npy')

labels = np.empty((numImages,10))
#for i in range(numImages):

numTrain = int(0.6*numImages)
numValid = int(0.15*numImages)
numTest = int(0.25*numImages)

trainCount = 0;
validCount = 0
testCount = 0;

classedImages = [[],[],[],[],[],[],[],[],[],[]]

traindata = []
validdata = []
testdata = []

trainlabels = []
validlabels = []
testlabels = []

for i in range(numImages):
    classedImages[l[i]].append(images[i])

for c in range(10):
    numThisClass = len(classedImages[c])
    trainCount = 0;
    validCount = 0
    testCount = 0;
    for i in range(numThisClass):
        
        if(trainCount < 0.6*numThisClass):
            traindata.append(classedImages[c][i])
            v = [0]*10
            v[c] = 1
            trainlabels.append(v)
            trainCount += 1
        elif(validCount < 0.15*numThisClass):
            validdata.append(classedImages[c][i])
            v = [0]*10
            v[c] = 1
            validlabels.append(v)
            validCount += 1
        else:
            testdata.append(classedImages[c][i])
            v = [0]*10
            v[c] = 1
            testlabels.append(v)
            testCount += 1
        

dt = tree.DecisionTreeClassifier()
dt = dt.fit(traindata, trainlabels)

predictions = dt.predict(validdata)

conf = np.zeros((10,10))

def getLabel(v):
    for i in range(10):
        if(v[i] == 1):
            return i

for i in range(numValid):
    reallabel = getLabel(validlabels[i])
    predictlabel = getLabel(predictions[i])
    conf[reallabel, predictlabel] += 1
