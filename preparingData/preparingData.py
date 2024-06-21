import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets
from tensorflow.keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten 
import cv2
import pickle
import os

# Load data
def load_Scale_data():
    # Load the MNIST dataset
    (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
    # Preprocess the data
    train_images = np.expand_dims(train_images, axis=-1).astype('float32') / 255.0
    test_images = np.expand_dims(test_images, axis=-1).astype('float32') / 255.0
    return train_images, test_images, train_labels, test_labels


# Visualize images
def showImages(num_images, data, label):
    for i in range(num_images):
        print(label[i])
        plt.figure(figsize=(7, 7))
        plt.imshow(data[i])
        plt.show()


# Resize grayscale images to RGB and preprocess for VGG16
def preprocess_images(images):
    processed_images = []
    for image in images:
        # Convert grayscale to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        # Resize to 48x48
        resized_image = cv2.resize(rgb_image, (48, 48))
        processed_images.append(resized_image)
    return preprocess_input(np.array(processed_images)) 


# Feature extraction using VGG16
def Feature_Model():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(48, 48, 3))  # Updated input_shape
    x = base_model.output
    x = Flatten()(x)
    feature_model = Model(inputs=base_model.input, outputs=x)
    return feature_model
# Feature extraction
def extract_features(feature_model, data):
    data_features = feature_model.predict(data)
    return data_features




if __name__ == "__main__":
    # Load and scale data and convert it into categories
    train_images, test_images, train_labels, test_labels = load_Scale_data()

    # Make preprocess for the train and test images
    train_images_resized = preprocess_images(train_images)
    test_images_resized = preprocess_images(test_images)
    # Extract features using the VGG16 model
    feature_model = Feature_Model()
    X_train_features = extract_features(feature_model, train_images_resized)
    X_test_features = extract_features(feature_model, test_images_resized)

    # Save all paths and its files
    paths_files = {'X_train_features.npy':X_train_features,
                    'X_test_features.npy':X_test_features,
                    'train_labels.pkl':train_labels,
                    'test_labels.pkl':test_labels}

    # Iterate on each file and save it if not exist
    for filename, data in paths_files.items():
        if not os.path.exists(filename):
            if filename.endswith('.npy'):
                np.save(filename, data)
            elif filename.endswith('.pkl'):
                with open(filename, 'wb') as f:
                    pickle.dump(data, f)

