from flask import Flask, request, jsonify
from flask_restful import Resource, Api
import cv2
import numpy as np
from joblib import load
import os

app = Flask(__name__)
api = Api(app)

classifier = load('dog_cat_svc.joblib')

def extract_features(image_path):
    img = cv2.imread(image_path)
    resized_img = cv2.resize(img, (100, 100))  # Resize image for consistency
    grayscale_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    flattened_img = grayscale_img.flatten()  # Flatten image into a 1D array
    return flattened_img

class PredictImage(Resource):
    def post(self):
        # Check if the POST request has the file part
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})

        file = request.files['file']

        # If the user does not select a file, the browser submits an empty file without a filename.
        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        if file:
            file_path = "uploaded_image.jpg"
            file.save(file_path)

            predicted_label = predict(file_path)

            return jsonify({'prediction': predicted_label})

def predict(path):
    flattened_test_image = extract_features(path)
    y_pred = classifier.predict(flattened_test_image.reshape(1, -1))
    labels = ['Cat', 'Dog']
    if y_pred == 0:
        return labels[0]
    if y_pred == 1:
        return labels[1]

api.add_resource(PredictImage, '/predict')

if __name__ == '__main__':
    app.run()