from flask import Flask, render_template, request, send_from_directory
import os
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from keras.utils import load_img, img_to_array
from keras.layers import GlobalMaxPooling2D
from sklearn.neighbors import NearestNeighbors
from product_display import get_random_products, get_product_details_by_id
from numpy.linalg import norm
import cv2

app = Flask(__name__)

# Set the UPLOAD_FOLDER
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Set the number of random products to display
NUM_PRODUCTS_TO_DISPLAY = 10

# Load pre-trained model and data
feature_list = np.array(pickle.load(open('xception_features.pkl', 'rb')))
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
    # Call the get_random_products function to get 20 random products
    random_products = get_random_products(num_products=NUM_PRODUCTS_TO_DISPLAY)

    # Extract product details from the DataFrame
    product_data = [
        {'productDisplayName': row['productDisplayName'], 'id': row['id'], 'paths': row['paths']} for _, row in
        random_products.iterrows()
    ]

    return render_template('index.html', products=product_data)

@app.route('/product/<int:product_id>')
def product_details(product_id):
    # Call the get_product_details_by_id function to get details for the main product
    main_product_details = get_product_details_by_id(product_id)

    # Extract features for the main product (assuming you have a function to extract features)
    main_product_features = feature_extraction(('static/' + main_product_details['paths']), model)

    # Use the features to recommend similar products
    indices = recommend(main_product_features, feature_list)
    similar_image_paths = [filenames[i] for i in indices[0][1:6]]

    # Fetch details for similar products
    similar_products_details = [
        get_product_details_by_id(int(image_path.split('/')[-1].split('.')[0])) for image_path in similar_image_paths
    ]

    return render_template('product_details.html',
                           product_details=main_product_details,
                           similar_products=similar_products_details)


if __name__ == '__main__':
    app.run(debug=True)
