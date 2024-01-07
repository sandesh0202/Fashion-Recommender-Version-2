from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from keras.utils import load_img, img_to_array
from keras.layers import GlobalMaxPooling2D
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import cv2
from product_display import get_random_products

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
NUM_PRODUCTS_TO_DISPLAY = 20

feature_list = np.array(pickle.load(open('xception_features.pkl', 'rb')))
filenames = np.array(pickle.load(open('filenamesPC.pkl', 'rb')))

base_model = tf.keras.applications.xception.Xception(weights='imagenet',
                      include_top=False,
                      input_shape=(224, 224, 3))
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    GlobalMaxPooling2D()
])

styles = pd.read_csv('static/myntradataset/new_styles.csv', on_bad_lines='skip')



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


@app.after_request
def add_security_headers(response):
    response.headers['Content-Security-Policy'] = "default-src 'self'; img-src 'self' data:"
    return response


@app.route('/', methods=['GET', 'POST'])
def index():
    unique_genders = styles['gender'].unique()
    unique_article_types = styles['articleType'].unique()

    if request.method == 'POST':
        selected_gender = request.form.get('gender', 'All')
        selected_article_type = request.form.get('article_type', 'All')

        index_products = get_random_products(gender=selected_gender, article_type=selected_article_type, num_products=NUM_PRODUCTS_TO_DISPLAY)
        product_data = [{'productDisplayName': row['productDisplayName'], 'id': row['id'], 'paths': row['paths']} for _, row in index_products.iterrows()]

        return render_template('index.html', products=product_data, unique_genders=unique_genders, unique_article_types=unique_article_types, selected_gender=selected_gender, selected_article_type=selected_article_type)

    return render_template('index.html', unique_genders=unique_genders, unique_article_types=unique_article_types)


@app.route('/product/<int:product_id>')
def product_details(product_id):
    main_product_details = styles[styles['id'] == product_id].iloc[0].to_dict()
    main_product_features = feature_extraction(('static/' + main_product_details['paths']), model)
    indices = recommend(main_product_features, feature_list)
    similar_image_paths = [filenames[i] for i in indices[0][1:6]]
    similar_products_details = [styles[styles['id'] == int(image_path.split('/')[-1].split('.')[0])].iloc[0].to_dict() for image_path in similar_image_paths]

    return render_template('product_details.html', product_details=main_product_details, similar_products=similar_products_details)


if __name__ == '__main__':
    app.run(debug=True)
