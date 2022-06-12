from __future__ import division, print_function
import sys
import os
import glob
import re
import numpy as np

import tensorflow as tf

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from keras.utils import image_utils


# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)

MODEL_PATH = 'recognition.h5'

model = load_model(MODEL_PATH)


print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    img =  image_utils.load_img(img_path, target_size=(224, 224))

    input_arr = tf.keras.preprocessing.image.img_to_array(img)
    input_arr = np.array([input_arr])  
    input_arr = input_arr.astype('float32') / 255

    predictions = model.predict(input_arr)

    pre_class=predictions.argmax()

    return pre_class


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':

        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)


        preds = model_predict(file_path, model)

        result = str(preds)

        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)