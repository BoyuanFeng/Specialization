from __future__ import print_function
import keras
from keras.datasets import cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import optimizers
import numpy as np
from keras.layers.core import Lambda
from keras import backend as K
from keras import regularizers
from keras.models import Model


import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.45
set_session(tf.Session(config=config))

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

        index_others = list(range(500))

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


        sp_y_train = y_train[train_index]
        sp_y_test = y_train[test_index]
        sp_x_train = x_train[train_index]
        sp_x_test = x_train[test_index]

        sp_x_train = sp_x_train.astype('float32')
        sp_x_test = sp_x_test.astype('float32')

        sp_y_train = keras.utils.to_categorical(sp_y_train, 11)
        sp_y_test = keras.utils.to_categorical(sp_y_test, 11)
        return sp_x_train, sp_x_test, sp_y_train, sp_y_test




class cifar100vgg:
    def __init__(self,train=True):
        self.num_classes = 100
        self.weight_decay = 0.0005
        self.x_shape = [32,32,3]

        self.model = self.build_model()
        if train:
            self.model = self.train(self.model)
        else:
            lrf = 0.1
            sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
            self.model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])
            self.model.load_weights('cifar100vgg.h5')
    def build_model(self):
        # Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.

        model = Sequential()
        weight_decay = self.weight_decay

        model.add(Conv2D(64, (3, 3), padding='same',
                         input_shape=self.x_shape,kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))

        model.add(Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))


        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))


        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(512,kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes))
        model.add(Activation('softmax'))
        return model


    def normalize(self,X_train,X_test):
        #this function normalize inputs for zero mean and unit variance
        # it is used when training a model.
        # Input: training set and test set
        # Output: normalized training set and test set according to the trianing set statistics.
        mean = np.mean(X_train,axis=(0,1,2,3))
        std = np.std(X_train, axis=(0, 1, 2, 3))
        print(mean)
        print(std)
        X_train = (X_train-mean)/(std+1e-7)
        X_test = (X_test-mean)/(std+1e-7)
        return X_train, X_test

    def normalize_production(self,x):
        #this function is used to normalize instances in production according to saved training set statistics
        # Input: X - a training set
        # Output X - a normalized training set according to normalization constants.

        #these values produced during first training and are general for the standard cifar10 training set normalization
        mean = 121.936
        std = 68.389
        return (x-mean)/(std+1e-7)

    def predict(self,x,normalize=True,batch_size=50):
        if normalize:
            x = self.normalize_production(x)
        return self.model.predict(x,batch_size)

    def train(self,model):

        #training parameters
        batch_size = 128
        maxepoches = 250
        learning_rate = 0.1
        lr_decay = 1e-6

        # The data, shuffled and split between train and test sets:
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train, x_test = self.normalize(x_train, x_test)

        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)

        lrf = learning_rate


        #data augmentation
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)



        #optimization details
        sgd = optimizers.SGD(lr=lrf, decay=lr_decay, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])


        # training process in a for loop with learning rate drop every 25 epoches.

        for epoch in range(1,maxepoches):

            if epoch%25==0 and epoch>0:
                lrf/=2
                sgd = optimizers.SGD(lr=lrf, decay=lr_decay, momentum=0.9, nesterov=True)
                model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

            historytemp = model.fit_generator(datagen.flow(x_train, y_train,
                                             batch_size=batch_size),
                                steps_per_epoch=x_train.shape[0] // batch_size,
                                epochs=epoch,
                                validation_data=(x_test, y_test),initial_epoch=epoch-1)
        model.save_weights('cifar100vgg.h5')
        return model




class half:
    def __init__(self,train=True):
        self.num_classes = 11
        self.weight_decay = 0.0005
        self.x_shape = [32,32,3]

        self.model = self.build_model()
        if train:
            self.model = self.train(self.model)
        else:
            lrf = 0.1
            sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
            self.model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])
            self.model.load_weights('half3.h5')
    def build_model(self):
        # Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.

        model = Sequential()
        weight_decay = self.weight_decay

        model.add(Conv2D(64, (3, 3), padding='same',
                         input_shape=self.x_shape,kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))

        model.add(Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))


        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(512,kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes))
        model.add(Activation('softmax'))
        return model


    def normalize(self,X_train,X_test):
        #this function normalize inputs for zero mean and unit variance
        # it is used when training a model.
        # Input: training set and test set
        # Output: normalized training set and test set according to the trianing set statistics.
        mean = np.mean(X_train,axis=(0,1,2,3))
        std = np.std(X_train, axis=(0, 1, 2, 3))
        print(mean)
        print(std)
        X_train = (X_train-mean)/(std+1e-7)
        X_test = (X_test-mean)/(std+1e-7)
        return X_train, X_test

    def normalize_production(self,x):
        #this function is used to normalize instances in production according to saved training set statistics
        # Input: X - a training set
        # Output X - a normalized training set according to normalization constants.

        #these values produced during first training and are general for the standard cifar10 training set normalization
        mean = 121.936
        std = 68.389
        return (x-mean)/(std+1e-7)

    def predict(self,x,normalize=True,batch_size=50):
        if normalize:
            x = self.normalize_production(x)
        return self.model.predict(x,batch_size)

    def distill1(self,y):
        result = []
        for i in range(y.shape[0]):
            sum = 0
            for j in range(10):
                sum += y[i][j]
            for j in range(10):
                result.append(y[i][j]*0.9/sum)
            result.append(0.1)
        result = np.array(result)
        result = result.reshape(y.shape[0], 11)
        print(type(result))
        print(result.shape)

        return result

        



    def train(self,model):

        #training parameters
        batch_size = 128
        maxepoches = 250
        learning_rate = 0.1
        lr_decay = 1e-6

        # The data, shuffled and split between train and test sets:


        x_train, x_test, y_train, y_test = generateSpecializedData()
        #(x_train, y_train), (x_test, y_test) = cifar100.load_data()

        originalModel = cifar100vgg(False)
        y_train = originalModel.model.predict(x_train)
        y_test = originalModel.model.predict(x_test)

        y_train = self.distill1(y_train)
        y_test  = self.distill1(y_test)

        lrf = learning_rate


        print(type(x_test))
        print(x_test.shape)    
        print(type(y_test))
        print(y_test.shape)
        print(type(x_train))
        print(x_train.shape)
        print(type(y_train))
        print(y_train.shape)
        
        #optimization details
        sgd = optimizers.SGD(lr=lrf, decay=lr_decay, momentum=0.9, nesterov=True)
        model.compile(loss='mean_squared_error', optimizer=sgd,metrics=['accuracy'])


        # training process in a for loop with learning rate drop every 25 epoches.

        for epoch in range(1,maxepoches):

            if epoch%25==0 and epoch>0:
                lrf/=2
                sgd = optimizers.SGD(lr=lrf, decay=lr_decay, momentum=0.9, nesterov=True)
                model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])

        historytemp = model.fit(x = x_train, y = y_train, batch_size = batch_size, epochs = epoch, verbose = 2, validation_data = (x_test, y_test))
        model.save_weights('half3.h5')
        return model

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    y_train = keras.utils.to_categorical(y_train, 100)
    y_test = keras.utils.to_categorical(y_test, 100)

    model = half()
    # using pre-trained model, the accuracy is 71%

    sp_x_train, sp_x_test, sp_y_train, sp_y_test = generateSpecializedData()


    #retrain_model = retrain(model, sp_x_train, sp_y_train)
    #predicted_x = retrain_model.predict(sp_x_test,50)
    predicted_x = model.predict(sp_x_test,50)
    residuals = (np.argmax(predicted_x,1)!=np.argmax(sp_y_test,1))
    loss = sum(residuals)/len(residuals)
    print("the validation 0/1 loss is: ",loss)

