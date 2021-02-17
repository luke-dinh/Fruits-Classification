import numpy as np
import keras
from keras.models import Model, load_model
from keras.layers import Conv2D, Flatten, Dense, Add, Input, ZeroPadding2D, AveragePooling2D, MaxPool2D
from keras.layers import BatchNormalization, Activation
import preprocessing
import matplotlib.pyplot as plt 
import time

from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

name = 'ResNet18-{}'.format(int(time.time()))
tensorboard = TensorBoard(log_dir='logs/{}'.format(name))

def id_block(X, filters):

    [F1,F2] = filters

    X_shortcut = X

    X = Conv2D(filters= filters[0], kernel_size=(3,3), strides=(1,1), padding='same')(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation(activation='relu')(X)

    X = Conv2D(filters=filters[1], kernel_size=(3,3), strides=(1,1), padding='same')(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation(activation='relu')(X)

    X = Add()([X_shortcut, X])
    X = Activation(activation='relu')(X)

    return X

def conv_block(X, filters):

    [F1,F2] = filters

    X_shortcut = X

    X = Conv2D(filters=filters[0], kernel_size=(3,3), strides=(1,1), padding='same')(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=filters[1], kernel_size=(3,3), padding='same', strides=(1,1))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    X_shortcut = Conv2D(filters=filters[1], kernel_size=(1,1), strides=(1,1), padding='valid')(X_shortcut)
    X_shortcut = BatchNormalization(axis=3)(X_shortcut)

    X = Add()([X_shortcut, X])
    X = Activation('relu')(X)

    return X

def resnet18(input_shape, classes=4):

    X_input = Input(shape=input_shape)
    X = ZeroPadding2D(padding=(2,2))(X_input)

    #1st stage
    X = Conv2D(filters=64, kernel_size=(7,7), strides=(2,2))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPool2D(pool_size=(3,3), strides=(2,2))(X)

    #2nd stage
    X = conv_block(X, filters=[64,64])
    X = id_block(X, filters=[64,64])

    #3rd stage
    X = conv_block(X, filters=[128,128])
    X = id_block(X, filters=[128,128])

    #4th stage
    X = conv_block(X, filters=[256,256])
    X = id_block(X, filters=[256,256])

    #5th stage
    X = conv_block(X, filters=[512,512])
    X = id_block(X, filters=[512,512])

    #6th layer
    X = AveragePooling2D(pool_size=(2,2))(X)
    X = Flatten()(X)
    X = Dense(units=1000)(X)
    X = Dense(units=classes, activation='softmax')(X)

    model = Model(inputs=X_input, outputs=X, name = 'ResNet18')

    return model

model = resnet18(input_shape=(100,100,3), classes=4)

datadir_train = 'Datasets/train'
datadir_val = 'Datasets/test'

train_generator = preprocessing.create_train(datadir=datadir_train)
val_generator = preprocessing.create_val(datadir_val=datadir_val)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

check_point = ModelCheckpoint('ResNet18.h5', monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early_stopping = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=3, verbose=1, mode='auto')

history = model.fit_generator(
    train_generator, validation_data=val_generator, epochs=10, verbose=1, steps_per_epoch=5, shuffle=True, callbacks=[check_point,early_stopping,tensorboard])

model.summary()
model.save('ResNet18.model')

keras.utils.plot_model(model, show_shapes=True)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.plot(acc, label='Training accuracy')
plt.plot(val_acc, label='Validation accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation accuracy')

plt.subplot(1,2,2)
plt.plot(loss, label='Training loss')
plt.plot(val_loss, label='Validation loss')
plt.legend(loc='upper right')
plt.title('Training and validation loss')
plt.show()
