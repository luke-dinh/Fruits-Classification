import tensorflow as tf 
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import os
import PIL
import numpy as np

categories = ["Apple", "Lemon", "Mango", "Raspberry"]

batch_size = 32
img_width = img_height = 100

# def modify(image):
#     contrast_image = ImageDataGenerator(brightness_range=[0.2,0.1])
#     return contrast_image

# def save_image(filename):
#     for category in categories:
#         file_path = os.oath.join(filename,category)
#         for img in os.listdir(file_path):
#             img = PIL.Image.open(img)
#             img = img_to_array(img)
#             img = modify(image=img)
#             img.save(os.path.join(img, "contrast" + str(i for i in len(os.listdir(file_path)))))
    
def create_train(datadir):
    train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=1, horizontal_flip=True, vertical_flip=True)
    train_generator = train_datagen.flow_from_directory(datadir, batch_size=32, class_mode='categorical', target_size=(img_width, img_height))
    return train_generator

def create_val(datadir_val):
    val_datagen = ImageDataGenerator(rescale=1./255)
    val_generator = val_datagen.flow_from_directory(datadir_val, class_mode='categorical', batch_size=32, target_size=(img_width, img_height))
    return val_generator

