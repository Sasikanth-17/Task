from flask import Flask, render_template, request, jsonify
import os
import cv2
import numpy as np
from sklearn.externals import joblib

app = Flask(__name__)

# Load the SVM model
svm_model = joblib.load('BrainTumorSVMModel.joblib')

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (64, 64))
    flattened_image = image.flatten()
    return flattened_image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', f.filename)
        f.save(file_path)

        # Preprocess the uploaded image
        processed_img = preprocess_image(file_path)

        # Make prediction using the SVM model
        prediction = svm_model.predict([processed_img])

        # Determine the result message
        result = "Yes Brain Tumor" if prediction[0] == 1 else "No Brain Tumor"

        return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
