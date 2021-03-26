# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 11:40:46 2021

@author: KrunalV
"""

from __future__ import division, print_function

import sys
import os
import glob
import re
import numpy as np
import cv2


# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField

# Define a flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY') or 'you-will-never-guess'
bootstrap = Bootstrap(app)

# Load your trained model
alex = tf.keras.models.load_model('alexnet_model.hdf5') 

labels = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 
            'Edema', 'Effusion', 'Emphysema', 
            'Fibrosis', 'Hernia', 'Infiltration', 
            'Mass', 'No Finding', 'Nodule', 
            'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']

print('Model loaded. Check http://127.0.0.1:5000/')

class UploadForm(FlaskForm):
    upload = FileField('Select an image:', validators=[
        FileRequired(),
        FileAllowed(['jpg', 'png', 'jpeg', 'JPEG', 'PNG', 'JPG'], 'Images only!')
    ])
    submit = SubmitField('Classify')

def get_prediction(img_path):

    pred_label = []
    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, (224, 224))
    img_preprocessed = preprocess_input(img_resized)
    img_reshaped = img_preprocessed.reshape((1, 224, 224, 3))
    prediction = alex.predict(img_reshaped)
    pred_class = prediction.argmax(axis=-1)
    pred_label.append(labels[pred_class[0]])
    print('PREDICTION LABEL:',pred_label)


    """
    print('IMAGE PATH: ', img_path)
    predict_datagen = ImageDataGenerator(rescale=1. / 255)
    predict = predict_datagen.flow_from_directory(
        'static/', 
        target_size=(224,224), 
        batch_size = 1,
        class_mode='categorical')
    pred = alex.predict_generator(predict)
    prediction = os.listdir(labels)[np.argmax(pred)]
    """
    return pred_label


@app.route('/', methods=['GET', 'POST'])
def home():
    form = UploadForm()
    if form.validate_on_submit():
        f = form.upload.data
        filename = secure_filename(f.filename)
        file_url = os.path.join('static/', filename)
        f.save(file_url)
        form = None
        prediction = get_prediction(file_url)
    else:
        file_url = None
        prediction = None
    return render_template("index.html", form=form, file_url=file_url, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)







"""

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x, mode='caffe')

    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, '', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        result = str(pred_class[0][0][1])               # Convert to string
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)

"""