from flask import Flask, request, redirect, url_for
from flask import send_from_directory
from numpy import loadtxt
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import tensorflow
import cv2
import os
import numpy as np

# load model
model = load_model('crop-disease.h5m')
# summarize model.
model.summary()

app = Flask(__name__)

@app.route('/')
def hello_world():
    return send_from_directory("static","index-redirect.html")

@app.route('/checkPicture', methods=['POST'])
def check_picture():
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        uploaded_file.save(uploaded_file.filename)
        image = get_data(uploaded_file.filename)
        print(image.shape)
        score = model.evaluate(image)

        print(score)
        print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))
        os.remove(uploaded_file.filename)
    return("");

def get_data(data_file):
    data = []
    img_width_size=512
    img_height_size=384
    #img_arr = cv2.imread(data_file)[...,::-1] #convert BGR to RGB format
    img_arr = cv2.imread(data_file,1)
    print(img_arr.shape)
    resized = cv2.resize(img_arr, (img_width_size, img_height_size),interpolation = cv2.INTER_AREA) # Reshaping images to preferred size
    resized = img_to_array(resized)
    #resized = np.expand_dims(resized, axis=0)  # this is creating tensor(4Dimension)
    print(resized.shape)
    #data.append([resized_arr, 1])
    return resized[None,:];

