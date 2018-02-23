###########
# IMPORTS #
###########
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.layers import AveragePooling2D
import math
import numpy

################
# PREPARATIONS #
################
# fix random seed for reproducibility
numpy.random.seed(7)
# load data set and test set (they have to have same dimensions)
dataset = numpy.loadtxt("ToricValues.txt", delimiter=",")
testset = numpy.loadtxt("ToricValues_test.txt", delimiter=",")
# note shape of indata
size_y = dataset.shape[0]
size_yt = testset.shape[0]
size_x = dataset.shape[1]
border = size_x - 4
# split into input (X) and output (Y) variables
X = dataset[:,0:border]
Y = dataset[:,border:size_x]
# do the same for test data
Xt = testset[:,0:border]
Yt = testset[:,border:size_x]
# convert input to matrices
N = int( -3/4 + math.sqrt(9/16 + (border - 1)/2) )
width = N + 1
height = 2*N + 1
X_i = numpy.empty([size_y, 1, width, height])
X_it = numpy.empty([size_yt, 1, width, height])
for j in range(width):
    for i in range(size_y):
        X_i[i,0,j,:] = X[i,j*height:(j+1)*height]
    for i in range(size_yt):
        X_it[i,0,j,:] = Xt[i,j*height:(j+1)*height]

#########
# MODEL #
#########
# Define model
model = Sequential()
model.add(Conv2D(15, kernel_size=[2, 3], input_shape=[width, height, 1], strides=[1, 2], padding='valid', activation='relu'))
model.add(AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid'))
model.add(Flatten())
model.add(Dense(N, activation='sigmoid'))
model.add(Dense(4, activation='softmax'))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

###################
# BACKPROPAGATION #
###################
# Fit the model
model.fit(X_i, Y, epochs=1000, batch_size=1000)

##############
# EVALUATION #
##############
# evaluate test data
scores = model.evaluate(X_it, Yt)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

###########
# PREDICT #
###########
# calculate predictions
predictions = model.predict(X_it)
# check accuracy
diff = numpy.rint(predictions) - Yt
res = numpy.empty([size_yt, 1], dtype='str')
for i in range(size_yt):
    res[i] = 'i'
    if numpy.argmax(predictions[i,:]) == numpy.argmax(Yt[i,:]):
        res[i] = 'c'

# print predictions
numpy.set_printoptions(threshold=numpy.inf)
print(numpy.concatenate([predictions, Yt, res],axis=1))
