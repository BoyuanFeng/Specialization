# Simple CNN model for CIFAR-10
import numpy
from keras.datasets import cifar10, cifar100
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
import numpy as np
K.set_image_dim_ordering('th')
from random import shuffle

import keras
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# 90%, 10 classes; 10%, others
def generateSpecializedData():
        (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')


        index0 = list(np.where(y_train[:,0] == 0)[0])
        index1 = list(np.where(y_train[:,0] == 1)[0])
        index2 = list(np.where(y_train[:,0] == 2)[0])
        index3 = list(np.where(y_train[:,0] == 3)[0])
        index4 = list(np.where(y_train[:,0] == 4)[0])
        index5 = list(np.where(y_train[:,0] == 5)[0])
        index6 = list(np.where(y_train[:,0] == 6)[0])
        index7 = list(np.where(y_train[:,0] == 7)[0])
        index8 = list(np.where(y_train[:,0] == 8)[0])
        index9 = list(np.where(y_train[:,0] == 9)[0])

        index_others = []
        for i in range(10000):
            if y_train[i,0] != 0 and y_train[i,0] != 1 and y_train[i,0] != 2 and y_train[i,0] != 3 and y_train[i,0] != 4 and y_train[i,0] != 5 and y_train[i,0] != 6 and y_train[i,0] != 7 and y_train[i,0] != 8 and y_train[i,0] != 9:
                index_others.append(i)
            if len(index_others) == 500:
                break

        train_index = index0[:450]
        train_index += index1[:450]
        train_index += index2[:450]
        train_index += index3[:450]
        train_index += index4[:450]
        train_index += index5[:450]
        train_index += index6[:450]
        train_index += index7[:450]
        train_index += index8[:450]
        train_index += index9[:450]
        train_index += index_others[:450]

        test_index = index0[-50:]
        test_index += index1[-50:]
        test_index += index2[-50:]
        test_index += index3[-50:]
        test_index += index4[-50:]
        test_index += index5[-50:]
        test_index += index6[-50:]
        test_index += index7[-50:]
        test_index += index8[-50:]
        test_index += index9[-50:]
        test_index += index_others[-50:]

        
        y_train[index_others] = 10


        shuffle(train_index)



        sp_y_train = y_train[train_index]
        sp_y_test = y_train[test_index]
        sp_x_train = x_train[train_index]
        sp_x_test = x_train[test_index]

        sp_x_train = sp_x_train.astype('float32')
        sp_x_test = sp_x_test.astype('float32')

        sp_y_train = keras.utils.to_categorical(sp_y_train, 11)
        sp_y_test = keras.utils.to_categorical(sp_y_test, 11)
        return sp_x_train, sp_x_test, sp_y_train, sp_y_test





# load data
X_train, X_test, y_train, y_test = generateSpecializedData()
# normalize inputs from 0-255 to 0.0-1.0
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

num_classes = y_test.shape[1]


# Create the model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(3, 32, 32), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
# Compile model
epochs = 50
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())

# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=32, verbose = 2)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=2)
print("Accuracy: %.2f%%" % (scores[1]*100))

predicted_x = model.predict(X_train,50)
residuals = (np.argmax(predicted_x,1)!=np.argmax(y_train,1))

predicted_x = model.predict(X_test,50)
residuals = (np.argmax(predicted_x,1)!=np.argmax(y_test,1))



loss = sum(residuals)/len(residuals)
print("the test accuracy is: ",(1-loss))





