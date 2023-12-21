from flask import Flask, render_template, request, send_from_directory
import os
import numpy as np
import pickle
import tensorflow as tf
from keras.applications import ResNet50
from keras.applications.imagenet_utils import preprocess_input
from keras.applications import ResNet50, Xception
from keras.utils import load_img, img_to_array
from keras.layers import GlobalMaxPooling2D
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
from PIL import Image

model = Xception(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

def xception_model(input_shape=(224, 224, 3)):
    # Load pre-trained Xception model
    base_model = Xception(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Freeze the weights of the pre-trained layers
    base_model.trainable = False
    
    # Build the Sequential model by adding Xception and GlobalMaxPooling2D layers
    model = tf.keras.Sequential([
        base_model,
        GlobalMaxPooling2D()
    ])
    
    return model

def feature_extraction(img_path, model):
    img = load_img(img_path, target_size=(224, 224, 3))
    img_array = img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result

def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    x, indices = neighbors.kneighbors([features])

    return indices