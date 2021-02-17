import keras
from keras.layers import Conv2D, Flatten, Dense, MaxPool2D
from keras.models import Sequential
from keras.callbacks import EarlyStopping, TensorBoard
import time
import matplotlib.pyplot as plt 
import preprocessing

name = 'model-{}'.format(int(time.time()))
tensorboard = TensorBoard(log_dir='logs/{}'.format(name))

model = Sequential()

model.add(Conv2D(16, kernel_size=(3,3),strides=(1,1), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(32, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(64, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(units=10))
model.add(Dense(units=4, activation='softmax'))

datadir_train = "Datasets/train"
datadir_val = "Datasets/test"

train_generator = preprocessing.create_train(datadir=datadir_train)
val_generator = preprocessing.create_val(datadir_val=datadir_val)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
early_stop = EarlyStopping(monitor='val_accuracy',min_delta=0,  patience=4, verbose=1, mode='auto')

history = model.fit_generator(train_generator, validation_data=val_generator, epochs=15, verbose=1, 
                                callbacks=[early_stop, tensorboard])

model.summary()
model.save('easymodel.model')

keras.utils.plot_model(model, show_shapes=True)

#Plot
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

# conv_layers = [1,2,3]
# num_filters = [8,16,32]
# dense_layers = [0,1,2]

# for dense_layer in dense_layers:
#     for num_filter in num_filters:
#         for conv_layer in conv_layers:

#             name = "{}-conv-{}-nodes-{}-dense".format(conv_layer,num_filter,dense_layer, int(time.time()))
#             model = Sequential()
#             model.add(Conv2D(num_filter, kernel_size=(3,3), padding='same', activation='relu'))
#             model.add(MaxPool2D(pool_size=(2,2)))
#             for l in range(conv_layer-1):
#                 model.add(Conv2D(num_filter, kernel_size=(3,3), padding='same', activation='relu'))
#                 model.add(MaxPool2D(pool_size=(2,2)))
#             model.add(Flatten())
#             for _ in range(dense_layer):
#                 model.add(Dense(num_filter, activation='elu'))
#             model.add(Dense(4, activation='softmax'))
#             tensor_board = TensorBoard(log_dir='log/{}'.format(name))
#             model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#             early_stopping = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=3, verbose=1, mode='auto')
#             model.fit_generator(train_generator, epochs=15, verbose=1, validation_data=val_generator, callbacks=[early_stopping, tensor_board])