from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn import datasets, svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import keras
import numpy as np

# Get Lists of each label from 0-9


images = np.load('images.npy')
labels = np.load('labels.npy')


flatImages = np.reshape(images, (6500, 28*28), order='C')

# Take the Training data and train the model

# Run it with the validation data. Keep going to see what might be the best.

# Can then add the validation data to training, make it one big training, and
# run the test data.


print(images.shape)
print(labels.shape)

print(flatImages.shape)

hotLabels = keras.utils.to_categorical(labels, num_classes=10)

index = np.array([i for i in range(6500)])

np.random.shuffle(index)

randImag = flatImages[index]
# randLabel = labels[index]

randLabelNonCat = labels[index]


# IMAGES

imag0 = randImag[randLabelNonCat == 0]
imag1 = randImag[randLabelNonCat == 1]
imag2 = randImag[randLabelNonCat == 2]
imag3 = randImag[randLabelNonCat == 3]
imag4 = randImag[randLabelNonCat == 4]
imag5 = randImag[randLabelNonCat == 5]
imag6 = randImag[randLabelNonCat == 6]
imag7 = randImag[randLabelNonCat == 7]
imag8 = randImag[randLabelNonCat == 8]
imag9 = randImag[randLabelNonCat == 9]

print("Images Shape")
print(imag0.shape)
print(imag1.shape)
print(imag2.shape)
print(imag3.shape)
print(imag4.shape)
print(imag5.shape)
print(imag6.shape)
print(imag7.shape)
print(imag8.shape)
print(imag9.shape)


ind01 = int(imag0.shape[0] * .6) - 1
ind02 = int(imag0.shape[0] * .75) - 1

ind11 = int(imag1.shape[0] * .6) - 1
ind12 = int(imag1.shape[0] * .75) - 1

ind21 = int(imag2.shape[0] * .6) - 1
ind22 = int(imag2.shape[0] * .75) - 1

ind31 = int(imag3.shape[0] * .6) - 1
ind32 = int(imag3.shape[0] * .75) - 1

ind41 = int(imag4.shape[0] * .6) - 1
ind42 = int(imag4.shape[0] * .75) - 1

ind51 = int(imag5.shape[0] * .6) - 1
ind52 = int(imag5.shape[0] * .75) - 1

ind61 = int(imag6.shape[0] * .6) - 1
ind62 = int(imag6.shape[0] * .75) - 1

ind71 = int(imag7.shape[0] * .6) - 1
ind72 = int(imag7.shape[0] * .75) - 1

ind81 = int(imag8.shape[0] * .6) - 1
ind82 = int(imag8.shape[0] * .75) - 1

ind91 = int(imag9.shape[0] * .6) - 1
ind92 = int(imag9.shape[0] * .75) - 1

training0 = imag0[:ind01]
training1 = imag1[:ind11]
training2 = imag2[:ind21]
training3 = imag3[:ind31]
training4 = imag4[:ind41]
training5 = imag5[:ind51]
training6 = imag6[:ind61]
training7 = imag7[:ind71]
training8 = imag8[:ind81]
training9 = imag9[:ind91]

print(ind51)
print(ind52)


validation0 = imag0[ind01 + 1: ind02]
validation1 = imag1[ind11 + 1: ind12]
validation2 = imag2[ind21 + 1: ind22]
validation3 = imag3[ind31 + 1: ind32]
validation4 = imag4[ind41 + 1: ind42]
validation5 = imag5[ind51 + 1: ind52]
validation6 = imag6[ind61 + 1: ind62]
validation7 = imag7[ind71 + 1: ind72]
validation8 = imag8[ind81 + 1: ind82]
validation9 = imag9[ind91 + 1: ind92]

print("Start Imag Valid")
print(validation0.shape)
print(validation1.shape)
print(validation2.shape)
print(validation3.shape)
print(validation4.shape)
print(validation5.shape)
print(validation6.shape)
print(validation7.shape)
print(validation8.shape)
print(validation9.shape)
print("End Imag Valid")

test0 = imag0[ind02 + 1:]
test1 = imag1[ind12 + 1:]
test2 = imag2[ind22 + 1:]
test3 = imag3[ind32 + 1:]
test4 = imag4[ind42 + 1:]
test5 = imag5[ind52 + 1:]
test6 = imag6[ind62 + 1:]
test7 = imag7[ind72 + 1:]
test8 = imag8[ind82 + 1:]
test9 = imag9[ind92 + 1:]


print("Start Image Test")
print(test0.shape)
print(test1.shape)
print(test2.shape)
print(test3.shape)
print(test4.shape)
print(test5.shape)
print(test6.shape)
print(test7.shape)
print(test8.shape)
print(test9.shape)
print("End Image Test")


# LABELS

labl0 = np.where(randLabelNonCat == 0)
labl1 = np.where(randLabelNonCat == 1)
labl2 = np.where(randLabelNonCat == 2)
labl3 = np.where(randLabelNonCat == 3)
labl4 = np.where(randLabelNonCat == 4)
labl5 = np.where(randLabelNonCat == 5)
labl6 = np.where(randLabelNonCat == 6)
labl7 = np.where(randLabelNonCat == 7)
labl8 = np.where(randLabelNonCat == 8)
labl9 = np.where(randLabelNonCat == 9)

print("Labels Shape")
print(labl0[0].shape)
print(labl1[0].shape)
print(labl2[0].shape)
print(labl3[0].shape)
print(labl4[0].shape)
print(labl5[0].shape)
print(labl6[0].shape)
print(labl7[0].shape)
print(labl8[0].shape)
print(labl9[0].shape)
print("Labels Shape End")

labl0_Loop = np.full((651, 1), 0)
labl1_Loop = np.full((728, 1), 1)
labl2_Loop = np.full((636, 1), 2)
labl3_Loop = np.full((669, 1), 3)
labl4_Loop = np.full((654, 1), 4)
labl5_Loop = np.full((568, 1), 5)
labl6_Loop = np.full((664, 1), 6)
labl7_Loop = np.full((686, 1), 7)
labl8_Loop = np.full((600, 1), 8)
labl9_Loop = np.full((644, 1), 9)

labl0_new = keras.utils.to_categorical(labl0_Loop, 10)
labl1_new = keras.utils.to_categorical(labl1_Loop, 10)
labl2_new = keras.utils.to_categorical(labl2_Loop, 10)
labl3_new = keras.utils.to_categorical(labl3_Loop, 10)
labl4_new = keras.utils.to_categorical(labl4_Loop, 10)
labl5_new = keras.utils.to_categorical(labl5_Loop, 10)
labl6_new = keras.utils.to_categorical(labl6_Loop, 10)
labl7_new = keras.utils.to_categorical(labl7_Loop, 10)
labl8_new = keras.utils.to_categorical(labl8_Loop, 10)
labl9_new = keras.utils.to_categorical(labl9_Loop, 10)

print("Lable 9 New Shape")
print(labl9_new.shape)
print(labl9_new[0])

indlab01 = int(labl0_new.shape[0] * .6) - 1
indlab02 = int(labl0_new.shape[0] * .75) - 1

indlab11 = int(labl1_new.shape[0] * .6) - 1
indlab12 = int(labl1_new.shape[0] * .75) - 1

indlab21 = int(labl2_new.shape[0] * .6) - 1
indlab22 = int(labl2_new.shape[0] * .75) - 1

indlab31 = int(labl3_new.shape[0] * .6) - 1
indlab32 = int(labl3_new.shape[0] * .75) - 1

indlab41 = int(labl4_new.shape[0] * .6) - 1
indlab42 = int(labl4_new.shape[0] * .75) - 1

indlab51 = int(labl5_new.shape[0] * .6) - 1
indlab52 = int(labl5_new.shape[0] * .75) - 1

indlab61 = int(labl6_new.shape[0] * .6) - 1
indlab62 = int(labl6_new.shape[0] * .75) - 1

indlab71 = int(labl7_new.shape[0] * .6) - 1
indlab72 = int(labl7_new.shape[0] * .75) - 1

indlab81 = int(labl8_new.shape[0] * .6) - 1
indlab82 = int(labl8_new.shape[0] * .75) - 1

indlab91 = int(labl9_new.shape[0] * .6) - 1
indlab92 = int(labl9_new.shape[0] * .75) - 1


trainlab0 = labl0_new[:ind01]
trainlab1 = labl1_new[:ind11]
trainlab2 = labl2_new[:ind21]
trainlab3 = labl3_new[:ind31]
trainlab4 = labl4_new[:ind41]
trainlab5 = labl5_new[:ind51]
trainlab6 = labl6_new[:ind61]
trainlab7 = labl7_new[:ind71]
trainlab8 = labl8_new[:ind81]
trainlab9 = labl9_new[:ind91]


validlab0 = labl0_new[ind01 + 1: ind02]
validlab1 = labl1_new[ind11 + 1: ind12]
validlab2 = labl2_new[ind21 + 1: ind22]
validlab3 = labl3_new[ind31 + 1: ind32]
validlab4 = labl4_new[ind41 + 1: ind42]
validlab5 = labl5_new[ind51 + 1: ind52]
validlab6 = labl6_new[ind61 + 1: ind62]
validlab7 = labl7_new[ind71 + 1: ind72]
validlab8 = labl8_new[ind81 + 1: ind82]
validlab9 = labl9_new[ind91 + 1: ind92]

print("Start Label")
print(validlab0.shape)
print(validlab1.shape)
print(validlab2.shape)
print(validlab3.shape)
print(validlab4.shape)
print(validlab5.shape)
print(validlab6.shape)
print(validlab7.shape)
print(validlab8.shape)
print(validlab9.shape)
print("End Label")

testlab0 = labl0_new[ind02 + 1:]
testlab1 = labl1_new[ind12 + 1:]
testlab2 = labl2_new[ind22 + 1:]
testlab3 = labl3_new[ind32 + 1:]
testlab4 = labl4_new[ind42 + 1:]
testlab5 = labl5_new[ind52 + 1:]
testlab6 = labl6_new[ind62 + 1:]
testlab7 = labl7_new[ind72 + 1:]
testlab8 = labl8_new[ind82 + 1:]
testlab9 = labl9_new[ind92 + 1:]

print("Start Label")
print(testlab0.shape)
print(testlab1.shape)
print(testlab2.shape)
print(testlab3.shape)
print(testlab4.shape)
print(testlab5.shape)
print(testlab6.shape)
print(testlab7.shape)
print(testlab8.shape)
print(testlab9.shape)
print("End Label")


trainingImag = np.vstack((training0, training1, training2, training3, training4,
                          training5, training6, training7, training8, training9))

validationImag = np.vstack((validation0, validation1, validation2, validation3, validation4,
                            validation5, validation6, validation7, validation8, validation9))


testImag = np.vstack((test0, test1, test2, test3, test4, test5, test6, test7, test8, test9))


trainingLabl = np.vstack((trainlab0, trainlab1, trainlab2, trainlab3, trainlab4,
                          trainlab5, trainlab6, trainlab7, trainlab8, trainlab9))

validationLabl = np.vstack((validlab0, validlab1, validlab2, validlab3, validlab4,
                            validlab5, validlab6, validlab7, validlab8, validlab9))


testLabl = np.vstack((testlab0, testlab1, testlab2, testlab3, testlab4,
                      testlab5, testlab6, testlab7, testlab8, testlab9))



print("Set Shapes")
print(trainingImag.shape)
print(validationImag.shape)
print(testImag.shape)
print(trainingLabl.shape)
print(validationLabl.shape)
print(testLabl.shape)
print("Set Shapes End")


# x[:4 Row, :5 Col]


# np.vstack takes a tuple of arrays and converts them into a single array. Use it on training, validation, and test.

# Look at Keras API in order to try and figure out how to do the model


# Error function. Take the derivative of the error function, and see how to minimize it.

# Model Template
model = Sequential()  # declare model
model.add(Dense(10, input_shape=(28*28, ), kernel_initializer='he_normal'))  # first layer
model.add(Activation('relu'))  # Non linear Activation

model.add(Dense(50, input_shape=(10, ), kernel_initializer='he_normal'))  # first layer
model.add(Activation('relu'))  # Non linear Activation

model.add(Dense(20, input_shape=(10, ), kernel_initializer='he_normal'))
model.add(Activation('relu'))  # Non linear Activation

# We have the model, the function that evaluates the model and we need kind of information we want it to know.
# Now we need to feed it data.

# We start with training. Check to see how model runs from Validation data. Run it again in Testing data.


#
#
#
# Fill in Model Here
#
#



model.add(Dense(10, input_shape=(10, ), kernel_initializer='he_normal'))
model.add(Activation('softmax'))


# Compile Model
model.compile(optimizer='sgd', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])


# Train Model
history = model.fit(trainingImag, trainingLabl, validation_data=(validationImag, validationLabl), epochs=100, batch_size=512)

# epocs: how many times the training is cycled. can be changed.
# batch size:

# Report Results
print(history.history)
predicted_lable=model.predict(testImag)  # Testing data goes in this function

len = len(testLabl)
print(len)



print(testLabl[0])
print(predicted_lable[0])
pred = predicted_lable.astype(int)

print(predicted_lable[0])
print("final")
print(np.argmax(predicted_lable,axis=1).shape)

pl=np.argmax(pred, axis=1)
tl=np.argmax(testLabl,axis=1)
print(np.argmax(testLabl,axis=1).shape)
matrix = confusion_matrix(pl, tl)

correct = accuracy_score(tl, pl)

print(matrix)
print(correct)
