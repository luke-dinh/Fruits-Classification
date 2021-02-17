from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.models import load_model
import tensorflow as tf
import cv2
import numpy as np

CATEGORIES = ["Apple", "Lemon", "Mango", "Raspberry"]
model = load_model("easymodel.model")
image = load_img("mango.jpg", target_size=(100,100))
image = img_to_array(image)
img_array = tf.expand_dims(image,0)
predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])
print("Class: {}".format(CATEGORIES[np.argmax(score)]))


#print to image
org = cv2.imread("mango.jpg")
cv2.putText(org, "Class: {}".format(CATEGORIES[np.argmax(score)]),
	(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
cv2.imshow("Classification", org)
cv2.waitKey(0)



