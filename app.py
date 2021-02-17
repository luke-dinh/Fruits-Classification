from __future__ import division, print_function
import os
import numpy as np
import cv2

import keras
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf 

import flask
from flask import Flask
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)

model_path = 'easymodel.model'
model = load_model(model_path)
print("Model was loaded. Check http://127.0.0.1:5000")

def predict(model, image):
    image = load_img(image, target_size=(100,100))
    img_array = img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)
    prediction = model.predict(img_array)
    score = tf.nn.softmax(prediction[0])

    return score

@app.route('/', methods= ['GET'])
def index():
    return flask.render_template("index.html")

@app.route('/predict', methods= ['GET','POST'])
def upload():
    if flask.request.method == 'POST':
        #get the file
        f = flask.request.files['file']
        #save to ./uploads'
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        # if not os.path.exists(file_path):
        #     os.makedirs(file_path)
        f.save(file_path)

        #predict
        pred = predict(model=model, image=file_path)

        CATEGORIES = ['Apple', 'Lemon', 'Mango', 'Raspberry']
        
        result = CATEGORIES[np.argmax(pred)]

        return result

    return None

if __name__ == '__main__':
    app.run(debug=True)
