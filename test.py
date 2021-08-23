from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.models import load_model
import tensorflow as tf
import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Test single image')

parser.add_argument("--image_path", defalut = "mango.jpg", type=str, help='path to test image')
parser.add_argument('--model_path', default= 'easymodel.model', type=str, help="path to model")
opt = parser.parse_args()

IMAGE_PATH = opt.image_path
MODEL_PATH = opt.model_path

CATEGORIES = ["Apple", "Lemon", "Mango", "Raspberry"]
model = load_model(MODEL_PATH)
image = load_img(IMAGE_PATH, target_size=(100,100))
image = img_to_array(image)
img_array = tf.expand_dims(image,0)
predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])
print("Class: {}".format(CATEGORIES[np.argmax(score)]))


#print to image
org = cv2.imread(IMAGE_PATH)
cv2.putText(org, "Class: {}".format(CATEGORIES[np.argmax(score)]),
	(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
cv2.imshow("Classification", org)
cv2.waitKey(0)



