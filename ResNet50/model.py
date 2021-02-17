import keras
from keras.layers import Conv2D, Dense, Flatten
from keras.layers import MaxPool2D, BatchNormalization
from keras.layers import Activation, Add, Input, ZeroPadding2D
from keras.models import Model, load_model
import preprocessing
import time
import matplotlib.pyplot as plt

from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

name = 'ResNet-{}'.format(int(time.time()))
tensorboard = TensorBoard(log_dir='logs/{}'.format(name))

def identity_block(X, filters, stage, block):

    #define name
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    [F1, F2, F3] = filters

    #Create shortcut
    X_shortcut = X

    #Define block:
    #1st 
    X = Conv2D(filters=filters[0], kernel_size=(1,1), strides=(1,1), padding='valid', name=conv_name_base + '2a')(X)
    X = BatchNormalization(axis=3, name=bn_name_base +'2a')(X)
    X = Activation(activation='relu')(X)

    #2nd
    X = Conv2D(filters=filters[1], kernel_size=(3,3), strides=(1,1), padding='same', name= conv_name_base + '2b')(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    #3rd
    X = Conv2D(filters=filters[2], kernel_size=(1,1), strides=(1,1), padding='valid', name=conv_name_base + '2c')(X)
    X = BatchNormalization(axis=3, name= bn_name_base + '2c')(X)

    #Add together
    X = Add()([X_shortcut, X])
    X = Activation('relu')(X)

    return X

def conv_block(X, filters, stage, block):

    #define name
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    #define variables
    [F1, F2, F3] = filters
    X_shortcut = X

    #layers
    #1st layer
    X = Conv2D(filters = filters[0], kernel_size=(1,1), strides=(2,2), padding='same', name=conv_name_base + '2a')(X)
    X = BatchNormalization(axis= 3, name= bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    #2nd layer
    X = Conv2D(filters= filters[1], kernel_size=(3,3), strides=(1,1), padding='same', name= conv_name_base + '2b')(X)
    X = BatchNormalization(axis= 3, name= bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    #3rd
    X = Conv2D(filters= filters[2], kernel_size=(1,1) ,strides=(1,1), padding='valid', name= conv_name_base + '2c')(X)
    X = BatchNormalization(axis=3, name= bn_name_base + '2c')(X)

    #Shortcut path with 1x1 conv layer
    X_shortcut = Conv2D(filters=filters[2], kernel_size=(1,1), strides=(2,2), padding='same', name= conv_name_base + '1')(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name= bn_name_base + '1')(X_shortcut)

    #Add together
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X

def resnet(input_shape=(100,100,3), classes=4):

    X_input = Input(shape= input_shape)
    X = ZeroPadding2D(padding=(3,3))(X_input)

    #1st stage
    X = Conv2D(filters=64, kernel_size=(7,7), strides=(2,2))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPool2D(pool_size=(2,2))(X)

    #2nd stage

    X = conv_block(X, filters= [64,64,256], stage=2, block= 'a')
    X = identity_block(X, filters= [64,64,256], stage=2, block='b')
    X = identity_block(X, filters= [64,64,256], stage=2, block='c')

    #3rd Stage
    X = conv_block(X, filters=[128,128,512], stage=3, block='a')
    X = identity_block(X, filters=[128,128,512], stage=3, block='b')
    X = identity_block(X, filters=[128,128,512], stage=3, block='c')
    X = identity_block(X, filters=[128,128,512], stage=3, block='d')

    #4th stage
    X = conv_block(X, filters=[256,256,1024], stage=4, block='a')
    X = identity_block(X, filters=[256,256,1024], stage=4, block='b')
    X = identity_block(X, filters=[256,256,1024], stage=4, block='c')
    X = identity_block(X, filters=[256,256,1024], stage=4, block='d')
    X = identity_block(X, filters=[256,256,1024], stage=4, block='e')
    X = identity_block(X, filters=[256,256,1024], stage=4, block='f')
    
    #5th stage
    X = conv_block(X, filters=[512, 512, 2048], stage=5, block='a')
    X = identity_block(X, filters= [512,512,2048], stage=5, block='b')
    X = identity_block(X, filters= [512,512,2048], stage=5, block='c')

    #6th layer
    X = MaxPool2D(pool_size=(2,2))(X)
    X = Flatten()(X)
    X = Dense(units=classes, activation='softmax')(X)

    model = Model(inputs=X_input, outputs=X, name='ResNet')

    return model

model = resnet(input_shape=(100,100,3), classes=4)

datadir_train = 'Datasets/train'
datadir_val = 'Datasets/test'

train_generator = preprocessing.create_train(datadir=datadir_train)
val_generator = preprocessing.create_val(datadir_val=datadir_val)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

check_point = ModelCheckpoint('ResNet.h5', monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early_stopping = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=3, verbose=1, mode='auto')

history = model.fit_generator(
    train_generator, validation_data=val_generator, epochs=10, verbose=1, steps_per_epoch=5, shuffle=True, callbacks=[check_point,early_stopping,tensorboard])

model.summary()
model.save('ResNet.model')

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
