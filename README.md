# Similar Product Recommender
This project implements an Similar Product Recommender using a Pre trained Xception model and Flask App. We leverages transfer learning with a pre-trained Xception model for feature extraction, which creates visual representation of images and k-Nearest Neighbors for finding similar images. Users can upload an image, and the system provides recommendations based on visual similarity.

This Recommendation System can be used by small scale E-commerce platforms who cannot afford to build recommendation system which uses user behaviour data for suggesting products. The Simplicity of this model makes it easy for small e-commerce industries to have Recommendation System that can Recommend Similar Products to users.

## Dataset 
We use ['Fashion Product Images (Small)'](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small/data) dataset which is a smaller version of Fashion Product Images Dataset which contains dataset scraped from Myntra. There are total 44,000 images and also You can also leverage styles.csv and styles.json file for using product text information.

## Feature Extraction

## Project Workflow
#### Image Upload
- The user accesses the web application through a browser and opens the main page.
- User selects an image file using the file input field on the Flask web interface.

#### Image Preprocessing
- Server creates a file path to save the uploaded image in the 'static/uploads' directory
- The uploaded image undergoes preprocessing, such as resizing and color channel adjustment, to match the model's input requirements.

#### Feature Extraction
- The server calls the 'feature_extraction' function, which uses a pre-trained Xception model for feature extraction.
- The Xception model extracts high-level features from the preprocessed image, generating a feature vector.
  
#### Find Similar Images
- The 'recommendation' function, which uses the Nearest Neighbors algorithm, is called.
- The algorithm searches for images in the pre-existing dataset with feature vectors most similar to the one extracted from the uploaded image.
  
#### Display Recommendations
- The uploaded image and the Recommended images file paths are passed to the 'result.html' template.
- The web interface displays the Uploaded image and a grid of visually Similar Recommended images.
  
  <img width="960" alt="Screenshot 2023-12-22 175249" src="https://github.com/sandesh0202/Fashion-Recommender-Version-2/assets/74035326/778ed2cd-3721-487b-b4a6-d880e48bf054">

## Feature Extraction 
For feature extraction we use a pre-trained [Xception model](https://www.tensorflow.org/api_docs/python/tf/keras/applications/xception/Xception). This function uses the process of Preparing an image, using a Pre-trained Xception model to Extract features, and Normalizing the resulting feature vector. The Normalized feature vector can then be used for similarity comparison with other images in the recommendation system.

## Usage 
- Clone - 
```
git clone https://github.com/sandesh0202/Fashion-Recommender-Version-2.git 
```
- Dataset -
  - Inside your project file create a folder named static
  - Extract you dataset in this file
    
- Feature Extraction process, (Check training.ipynb file)
  - Run training.ipynb step by step for Feature Extraction of all images in dataset
  - Adjust file paths and names of files according to your need in this notebook
  Note - If your PC is not able to do the feature extraction process, run the same notebook on Google Colab or Kaggle, download the pickle file and save that pickle file inside your project 

- app.py
  - Optimize all file paths according to your file paths
  - run app.py
  ```
    python app.py
  ```

# ThankYou, Project by [@Sandesh](https://github.com/sandesh0202)
 
