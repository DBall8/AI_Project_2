import numpy as np
from sklearn import tree
import graphviz as gp


def getLabel(v):
    for i in range(10):
        if(v[i] == 1):
            return i

def getPandR(conf):
    recall = np.zeros((10))
    precision = np.zeros((10))
    for c in range(10):
        TP = conf[c,c]
        FN = 0
        FP = 0
        for i in range(10):
            if i != c:
                FN += conf[c,i]
        for i in range(10):
            if i != c:
                FP += conf[i,c]
        recall[c] = TP / (TP + FN)
        precision[c] = TP / (TP + FP)
    return precision, recall

def compare(v1, v2):
    v3 = np.zeros((10))
    for i in range(10):
        v3[i] = v1[i] - v2[i]
    return v3

def getConfMatrix(predictions, validlabels):
    conf = np.zeros((10,10))
    for i in range(len(validlabels)):
        reallabel = getLabel(validlabels[i])
        predictlabel = getLabel(predictions[i])
        conf[reallabel, predictlabel] += 1
    return conf

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
        
def makeTree(testdata, testlabels):
    global traindata, trainlabels
    dt = tree.DecisionTreeClassifier(max_depth=20)
    dt = dt.fit(traindata, trainlabels)
    predictions = dt.predict(testdata)
    conf = getConfMatrix(predictions, testlabels)
    return conf

def saveData(conf):
    file = open('confusion_matrix.csv', "w")
    for r in conf:
        for i in r:
            file.write(str(i) + ',')
        file.write('\n')
    file.close()

    p, r = getPandR(conf)

    file = open('precision.csv', "w")
    for i in p:
        file.write(str(i) + ',')
    file.close()

    file = open('recall.csv', "w")
    for i in r:
        file.write(str(i) + ',')
    file.close()

#precision, recall = getPandR(conf)

##print('Precision')
##print(precision)
##print('Recall')
##print(recall)
##
##print('Precision2')
##print(precision)
##print('Recall2')
##print(recall)
#print(conf)


##print("Difference")
##print('Precision')
##print(compare(precision, precision2))
##print('Recall')
##print(compare(recall, recall2))




