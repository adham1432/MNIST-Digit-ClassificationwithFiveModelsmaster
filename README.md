# MNIST Classification Project Overview
This project focuses on classifying handwritten digits from the MNIST dataset using various machine learning models. The models implemented include a Deep Neural Network (DNN), Decision Tree, Random Forest, Naive Bayes, and K Nearest Neighbors (KNN) classifiers.

![Image about the final project](<mnist classifier with five models.png>)

## Prerequisites
To run the project, you'll need the following libraries:
- NumPy
- Flask
- scikit-learn
- Keras
- OpenCV (cv2)
- joblib
- matplotlib

## Overview of the Code
The project consists of the following components:
- **Model Training (create_model.py):** This file contains the code to train the machine learning models on the MNIST dataset. Models include Decision Tree, Random Forest, DNN, Naive Bayes, and KNN classifiers. The trained models are then saved for later use.
- **Model Deployment (app.py):** This script creates a Flask web application to deploy the trained models. It loads the trained models and provides a web interface for users to upload images of handwritten digits and get predictions from the models.
- **HTML Templates (home.html):** This HTML file defines the structure of the web interface. It includes an input form for selecting the model and uploading images, and it displays the prediction results.
- **CSS Styles (home.css):** This CSS file contains styles to enhance the appearance of the web interface.

## Models Accuracy
- **Deep Neural Network (DNN):** Achieved an accuracy of 82% on the test set.
- **Decision Tree (DT):** Achieved an accuracy of 76% on the test set.
- **Random Forest (RF):** Achieved an accuracy of 87% on the test set.
- **Naive Bayesian (NB):** Achieved an accuracy of 61% on the test set.
- **K Nearest Neighbors (KNN):** Achieved an accuracy of 87% on the test set.

## Flask App Structure
- **app.py:** Contains Flask routes for rendering the web interface and handling predictions.
- **templates/:** Directory with HTML templates for the web pages.
- **static/:** Directory for static files as CSS.

## Contribution
Contributions to this project are welcome. You can help improve the model's accuracy, explore different DNN architectures, or enhance the data preprocessing and visualization steps. Feel free to make any contributions and submit pull requests.
