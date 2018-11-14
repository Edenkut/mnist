#Keras is a high-level neural network API. Keras is written in pure Python and based on Tensorflow, Theano and CNTK backends.
#About this research,I used a convolutional neural network structure to test the mnist dataset.


#1-CNN example

#
import keras
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import RMSprop

#loading data
(x_train, y_train), (x_test, y_test) = mnist.load_data()


#Randomly divide the matrix into training subsets and test subsets
x_train1, x_valid, y_train1, y_valid = train_test_split(x_train, y_train, test_size=0.175)

# data pre-processing
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_valid = x_valid.reshape(x_valid.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

x_train = x_train.astype('float32')
x_valid = x_valid.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_valid /= 255
x_test /= 255

#exchange to Binary category matrix
y_train = keras.utils.to_categorical(y_train, 10)
y_valid = keras.utils.to_categorical(y_valid, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# begin to build model
model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

#define Losee func
model.compile(loss='categorical_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])

#Evaluation model
history = model.fit(x_train, y_train,batch_size=128,epochs=10,verbose=1,validation_data=(x_valid, y_valid))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
