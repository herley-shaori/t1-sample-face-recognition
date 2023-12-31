# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.
from __future__ import print_function
from flask import request
import os
import pickle
import boto3
import flask
import cv2
import io
import pandas as pd
import numpy as np
import base64
from PIL import Image

prefix = "/opt/ml/"
model_path = os.path.join(prefix, "model")


# The flask app for serving predictions
app = flask.Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'inference_input'
cascade_path = "haarcascade_frontalface_default.xml"  # Path to face cascade file
face_cascade = cv2.CascadeClassifier(cascade_path)

# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.


class ScoringService(object):
    s3 = boto3.client('s3')
    s3.download_file('temp-data-for-rekognition', 'byom-model/model.pkl', 'model.pkl')
    model = 'model.pkl'
    @classmethod
    def get_model(cls):
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

@app.route("/ping", methods=["GET"])
def ping():
    health = ScoringService.get_model() is not None
    status = 200 if health else 404
    return flask.Response(response="\n", status=status, mimetype="application/json")

@app.route("/invocations", methods=["POST"])
def transformation():
    if flask.request.content_type == "text/csv":
        # data = flask.request.data.decode("utf-8")
        # s = io.StringIO(data)
        # df = pd.read_csv(s, header=None)
        # payload = df.values[0][0]
        # print(payload)
        with open('encode.bin', "wb") as file:
            file.write(flask.request.data)
        file = open('encode.bin', 'rb')
        byte = file.read()
        file.close()
        filepath = 'saved_payload.jpg'
        decodeit = open(filepath, 'wb')
        decodeit.write(base64.b64decode((byte)))
        decodeit.close()
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
        return "OK"
    else:
        return flask.Response(
            response="This predictor only supports CSV data", status=415, mimetype="text/plain"
        )
        return "OK"
    return "Tidak OK:"+flask.request.content_type

if __name__ == "__main__":
    app.run()