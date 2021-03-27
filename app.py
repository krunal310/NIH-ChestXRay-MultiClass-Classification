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
import json
from flask import Flask, redirect, url_for, request, render_template, jsonify, make_response
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField



#addition by @tuminzee
from pymongo import MongoClient
from bson.objectid import ObjectId
from skimage import io
# import urllib
# from skimage import io

# Define a flask app
app = Flask(__name__)
client = MongoClient('localhost', 27017)
db = client['SehatIntel']
datas_collection = db['datas']
users_collection = db['users']


# app.config['MONGODB_SETTINGS'] = {
#     'db': 'SehatIntel',
#     'host': 'localhost',
#     'port': 27017
# }

# db = MongoEngine()
# db.init_app(app)
# class User(db.Document):
#     name = db.StringField()
#     email = db.StringField()

# class Users(db.Document):
#     picture = db.StringField()
#     name = db.StringField()

# class Datas(db.Document):
#     name: db.StringField()
#     age: db.IntField()
#     diseasesName: db.StringField()
#     doctorsFeedback: db.StringField()
#     doctorsName: db.StringField()
#     email: db.StringField()
#     hospitalsName: db.StringField()
#     prescription: db.StringField()
#     labReportFileId: db.StringField()
#     labReportFileUrl: db.StringField()
#     caseStartDate: db.DateTimeField()
#     caseEndDate: db.DateTimeField()



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



# def url_to_image(url):
# 	# download the image, convert it to a NumPy array, and then read
# 	# it into OpenCV format
# 	resp = urllib.urlopen(url)
# 	image = np.asarray(bytearray(resp.read()), dtype="uint8")
# 	image = cv2.imdecode(image, cv2.IMREAD_COLOR)
# 	# return the image
# 	return image


def get_prediction(img_path):

    pred_label = []
    # img = cv2.imread(img_path)
    # img = url_to_image(img_path)
    img = io.imread(img_path)
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


@app.route('/image', methods=['POST', 'GET'])
def pymongo_test():
    if request.method == 'POST':
        request_data = request.get_json()
        user_data = datas_collection.find_one({'_id': ObjectId(request_data['id'])})
        if( user_data):
            prediction = get_prediction(user_data['labReportFileUrl'])
            filter = {
                '_id':  ObjectId(request_data['id'])
            }
            new_values = {
                "$set": {
                    'labReportDiagnosistics': str(prediction)
                }
            }
            datas_collection.update_one(filter, new_values)
            return "data updated"
        else:
            return 'Try again with different object id'
    else:
        data = []
        cursor  = datas_collection.find()
        for doc in cursor:
            doc['_id'] = str(doc['_id'])
            data.append(doc)
        return jsonify(data)



# @app.route('/mongo', methods=['GET', 'POST'])
# def mongodb_test():
#     if request.method == 'GET':
#         users = Users.objects()
#         return (jsonify(users), 200)

#     else:
#         # record = json.loads(request.data)
#         body = request.get_json()
#         # return body
#         user = User.objects(name=body['name']).first()
#         if not user:
#             return jsonify({'error': 'data not found'})
#         else:
#             user.update(email=body['email'])
#         return jsonify(user.to_json())

# @app.route('/image', methods=['GET', 'POST'])
# def image_predict_from_mongodb():
    # if request.method == 'POST':
    #     request_data = request.get_json()
    #     user_data = Datas.objects(_id=request_data['id']).first()
    #     if not user_data:
    #         return jsonify({'error': 'data not found'})
    #     else:
    #        return 'try again'
    #     return jsonify(user.to_json())


if __name__ == "__main__":
    app.run(debug=True)


