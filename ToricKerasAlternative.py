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

# load training set
horset  = numpy.loadtxt("ToricValues_horiz.txt",delimiter=",")
vertset = numpy.loadtxt("ToricValues_vert.txt",delimiter=",")

# load test set
horsettest=numpy.loadtxt("ToricValues_horiz_test.txt",delimiter=",")
vertsettest=numpy.loadtxt("ToricValues_vert_test.txt",delimiter=",")

# extract dimensions
size_x=horset.shape[1]
size_y=horset.shape[0]
size_yt=horsettest.shape[0]
border=size_x-4

# create X and Y for training
Xhor=horset[:,0:border]
Xvert=vertset[:,0:border]
Y=horset[:,border:size_x]

# create X and Y for testing
Xthor=horsettest[:,0:border]
Xtvert=vertsettest[:,0:border]
Yt=horsettest[:,border:size_x]

# matrix dimensionality
length = int(math.sqrt(border))

# define input sets
X_i  = numpy.empty([size_y,2,length,length])
X_it = numpy.empty([size_yt,2,length,length])

# construct input sets
for j in range(length):
    for i in range(size_y):
        X_i[i,0,j,:] = Xhor[i,j*length:(j+1)*length]
        X_i[i,1,:,j] = Xvert[i,j*length:(j+1)*length]
    for i in range(size_yt):
        X_it[i,0,j,:] = Xthor[i,j*length:(j+1)*length]
        X_it[i,1,:,j] = Xtvert[i,j*length:(j+1)*length]

#########
# MODEL #
#########

# Define model
model = Sequential()
model.add(Conv2D(1, kernel_size=[1, length], input_shape=[2, length, length], strides=1, padding='valid', activation='relu', data_format='channels_first'))
#model.add(AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid'))
model.add(Flatten())
model.add(Dense(border, activation='sigmoid'))
model.add(Dense(4, activation='sigmoid'))

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

###################
# BACKPROPAGATION #
###################
# Fit the model
cb = TensorBoard()
model.fit(X_i, Y, epochs=100, batch_size=100, verbose=1, callbacks=[cb])

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
