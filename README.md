# Product Recommendation System

## Business Understanding

### Project Overview
This project aims to implement a Similar Product Recommender using a pre-trained Xception model and a Flask web application. The system leverages transfer learning to extract visual features from images and employs k-Nearest Neighbors for finding similar images. The target audience for this Recommender System includes small-scale e-commerce platforms that may not have the resources for complex recommendation systems based on user behavior data.

### Objectives
- Implement a user-friendly web application for image upload.
- Utilize transfer learning with a pre-trained Xception model for feature extraction.
- Employ k-Nearest Neighbors algorithm for recommending visually similar products.
- Provide recommendations based on visual similarity rather than user behavior data.

## Data Understanding

### Dataset
The project uses the ['Fashion Product Images (Small)'](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small/data) dataset, a subset of Fashion Product Images scraped from Myntra. The dataset consists of 44,000 images, and additional information is available in styles.csv and styles.json files.

## Project Workflow

### Image Upload
1. Users access the web application through a browser and open the main page.
2. Users select an image file using the file input field on the Flask web interface.

### Image Preprocessing
1. The server creates a file path to save the uploaded image in the 'static/uploads' directory.
2. The uploaded image undergoes preprocessing, including resizing and color channel adjustment, to match the model's input requirements.

### Feature Extraction
1. The server calls the 'feature_extraction' function, utilizing a pre-trained Xception model.
2. The Xception model extracts high-level features from the preprocessed image, generating a feature vector.

### Find Similar Images
1. The 'recommendation' function is called, employing the Nearest Neighbors algorithm.
2. The algorithm searches for images in the pre-existing dataset with feature vectors most similar to the one extracted from the uploaded image.

### Display Recommendations
1. The uploaded image and the recommended image file paths are passed to the 'result.html' template.
2. The web interface displays the uploaded image and a grid of visually similar recommended images.

## Feature Extraction

### Xception Model
For feature extraction, a pre-trained [Xception model](https://www.tensorflow.org/api_docs/python/tf/keras/applications/xception/Xception) is utilized. The process involves:
1. **Image Preparation:** The uploaded image is prepared for analysis.
2. **Feature Extraction:** The Xception model is used to extract features from the image.
3. **Normalization:** The resulting feature vector is normalized for comparison with other images in the recommendation system.

### Representation as a Feature Vector
Feature extraction involves using the Xception model to identify significant visual patterns and characteristics within an uploaded image. The resulting feature vector represents a high-level abstraction of the image's visual content. By reducing the dimensionality of the data while retaining essential information, the feature vector enables effective similarity comparison with other images in the recommendation system.

## Index Page Documentation

### Overview

The index page serves as the main entry point for users to explore a random selection of products from the dataset. Users can also apply filters based on gender and article type to customize the displayed products.

### Usage

#### 1. Navigation

- The main navigation menu contains options to filter products based on gender and article type.
- Users can select specific filters or leave them as 'All' to view a diverse set of products.

#### 2. Product Display

- The page displays a set of product cards, each showcasing an image and essential details of a randomly selected product.
- Clicking on a product card will lead users to the product details page.

#### 3. Filter Options

- Users can choose to filter products by gender and article type using the provided dropdown menus.
- Changing filter options will dynamically update the displayed products.

#### 4. Reset Filters

- To reset filters and display a random set of products, users can click on the 'Reset Filters' button.

### Examples

#### Filtering by Gender and Article Type

1. Select 'Men' from the gender dropdown.
2. Choose 'Shirts' from the article type dropdown.
3. The displayed products will now be filtered accordingly.

#### Resetting Filters

1. Click on the 'Reset Filters' button.
2. The page will reload, showing a new set of random products without any specific filters.

### Note

- The displayed products are sampled randomly, ensuring a diverse and dynamic user experience.
- The filters provide users with the flexibility to explore products tailored to their preferences.

## Product Details Page Documentation

### Overview

The product details page provides in-depth information about a selected product, including its image, name, and additional details. Users can also explore similar products based on features extracted from the main product.

### Usage

#### 1. Main Product Information

- The page prominently displays the main product's image and name.
- Users can review detailed information about the selected product.

#### 2. Similar Products Section

- The "Similar Products" section showcases a set of products that share similarities with the main product.
- Each similar product is presented with an image, name, and other relevant details.

#### 3. Clickable Product Cards

- Both the main product card and similar product cards are clickable.
- Clicking on any product card will navigate users to the respective product details page.

#### 4. Navigation

- The "Back to Recommendations" link allows users to return to the index page and explore more product recommendations.

### Examples

#### Exploring Similar Products

1. Open a product details page.
2. Scroll down to the "Similar Products" section.
3. Click on any similar product card to view its details.

#### Returning to Recommendations

1. Click on the "Back to Recommendations" link.
2. The page will redirect to the index page, providing a fresh set of product recommendations.

### Note

- The product details page leverages features extracted from the main product to recommend similar products.
- Clickable product cards enhance user interaction and navigation between different product details pages.


## Steps
- Clone - 
```
git clone https://github.com/sandesh0202/Fashion-Recommender-Version-2.git 
```
- Dataset -
  - Inside your project file create a folder named static
  - Extract your dataset in this file
    
- Feature Extraction process, (Check training.ipynb file)
  - Run training.ipynb step by step for Feature Extraction of all images in dataset
  - Adjust file paths and names of files according to your need in this notebook
  Note - If your PC is not able to do the feature extraction process, run the same notebook on Google Colab or Kaggle, download the pickle file and save that pickle file inside your project 

- node js integration
  - Install node.js in your file, it will install all required files
  ```
    npm install
  ```
  
- app.py
  - Optimize all file paths according to your file paths
  - run app.py
  ```
    python app.py
  ```

## ThankYou!
 
---
