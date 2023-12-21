from flask import Flask, render_template, request, send_from_directory
import os
import numpy as np
import pickle
import tensorflow as tf
from keras.applications import ResNet50, Xception
from keras.applications.imagenet_utils import preprocess_input
from keras.utils import load_img, img_to_array
from keras.layers import GlobalMaxPooling2D
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
from PIL import Image
from sklearn.decomposition import PCA
import cv2


app = Flask(__name__)


# Load pre-trained model and data
feature_list = np.array(pickle.load(open('xception_pca.pkl', 'rb')))
filenames = np.array(pickle.load(open('filenamesPC.pkl', 'rb')))

# Pre trained model
base_model = tf.keras.applications.xception.Xception(weights='imagenet',
                      include_top=False,
                      input_shape = (224, 224, 3))
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    GlobalMaxPooling2D()
])

# Feature Extraction
def feature_extraction(img_path, model):
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    
    img_array_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    expanded_img_array = np.expand_dims(img_array_bgr, axis=0)
    preprocessed_img = tf.keras.applications.xception.preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result

def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    x, indices = neighbors.kneighbors([features])

    return indices

# Add a Content Security Policy (CSP) header
@app.after_request
def add_security_headers(response):
    response.headers['Content-Security-Policy'] = "default-src 'self'; img-src 'self' data:"
    return response

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return render_template('index.html', error="No file part")

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', error="No selected file")

    try:
        os.makedirs('static/uploads', exist_ok=True)
        file_path = os.path.join('static/uploads', file.filename)
        file.save(file_path)

        features = feature_extraction(file_path, model)
        indices = recommend(features, feature_list)

        # Exclude the first index and get the top 6 recommended image paths
        top6_image_paths = [filenames[i] for i in indices[0][1:6]]
        print("Rendering images:", top6_image_paths)  # Print for debugging

        # Pass uploaded image path and top 6 image paths to the template
        return render_template('result.html', uploaded_image_path=file.filename, image_paths=top6_image_paths)

    except Exception as e:
        print(f"Error processing file: {e}")
        return render_template('index.html', error="Error processing file")


if __name__ == '__main__':
    app.run(debug=True)
