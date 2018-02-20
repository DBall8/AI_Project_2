import numpy as np
from sklearn import tree
import graphviz

images = np.load('images.npy')
d = np.shape(images)
numImages = d[0]
images = np.reshape(images, (numImages, 784))

l = np.load('labels.npy')

labels = np.empty((numImages,10))
for i in range(numImages):
    v = np.zeros((10))
    v[l[i]] = 1
    labels[i] = v

classes = np.empty((0))

dt = tree.DecisionTreeClassifier()
dt = dt.fit(images, labels)

predictions = dt.predict(images)

#data = tree.export_graphviz(dt, out_file=None)

#graph = graphviz.Source(data)
#graph.render("dt")

