# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.
from __future__ import print_function
from flask import request
import os
import pickle
import boto3
import flask
import cv2

import numpy as np

prefix = "/opt/ml/"
model_path = os.path.join(prefix, "model")

# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.


class ScoringService(object):
    s3 = boto3.client('s3')
    s3.download_file('temp-data-for-rekognition', 'byom-model/model.pkl', 'model.pkl')
    # os.system('cp model.pkl /opt/ml/model/model.pkl')
    model = 'model.pkl'

    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if(os.path.exists('model.pkl')):
            with open("model.pkl", "rb") as inp:
                cls.model = pickle.load(inp)
        if (cls.model == None):
            with open(os.path.join(model_path, "model.pkl"), "rb") as inp:
                cls.model = pickle.load(inp)
        return cls.model

    @classmethod
    def predict(cls, input):
        clf = cls.get_model()
        return clf.predict(input)


# The flask app for serving predictions
app = flask.Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'inference_input'
cascade_path = "haarcascade_frontalface_default.xml"  # Path to face cascade file
face_cascade = cv2.CascadeClassifier(cascade_path)

@app.route("/ping", methods=["GET"])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = ScoringService.get_model() is not None  # You can insert a health check here
    status = 200 if health else 404
    return flask.Response(response="\n", status=status, mimetype="application/json")

@app.route("/invocations", methods=["POST"])
def transformation():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    image = request.files['image']
    print(image)
    if (image):
        filename = image.filename
        filepath = app.config['UPLOAD_FOLDER'] + '/' + filename
        image.save(filepath)
        img = cv2.imread(filepath)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cropped_img = img[y:y + h, x:x + w]
            img = cropped_img.flatten()
            if(len(img) < 56307):
                img = np.pad(img, (0, 56307 - len(img)), mode='constant')
            img = img.reshape(1,-1)
            result = ScoringService.predict(img)
            return flask.Response(response=result, status=200)
    return flask.Response(response='No face detected!', status=200)