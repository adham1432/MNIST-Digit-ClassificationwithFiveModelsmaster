# Import required libraries
from preparingData.preparingData import Feature_Model, extract_features, preprocess_images
from flask import Flask, render_template, request
from keras.models import load_model
import numpy as np
import base64
import cv2
import joblib

# Load models
dnn_model = load_model('Models\dnn_model.h5')
dt_model = joblib.load('Models\decision_tree_model.joblib')
rf_model = joblib.load('Models\\random_forest_model.joblib')
nb_model = joblib.load('Models\\naiveclassifier.joblib')
knn_model = joblib.load('Models\knn.joblib')
# Create feature model
feature_model = Feature_Model()

# Create application
app = Flask(__name__, template_folder="templates", static_folder="static", static_url_path="/")

# Make home route
@app.route('/')
def index():
    return render_template('home.html', prediction = False)

# Make predict route
@app.route('/predict',methods = ['POST'])
def predict():
    # Retrieve selected model
    selected_model = request.form['model']
    # Retrieve uploaded image
    uploaded_image = request.files['image']
    # Read and decode image
    img_array = np.frombuffer(uploaded_image.read(), np.uint8)
    img_cv2 = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
    # Scale image
    scaled_image = img_cv2.astype('float32') / 255.0
    # Preprocess image for VGG16
    preprocessed_image = preprocess_images([scaled_image])
    # Extract image features
    image_features = extract_features(feature_model, preprocessed_image)
    # Check on the model type
    if selected_model == "dnn_model":
        predicted_class = np.argmax(dnn_model.predict(image_features))
    elif selected_model == "dt_model":
        predicted_class = dt_model.predict(image_features)[0]
    elif selected_model == "nb_model":
        predicted_class = nb_model.predict(image_features)[0]
    elif selected_model == "knn_model":
        predicted_class = knn_model.predict(image_features)[0]
    else:
        predicted_class = rf_model.predict(image_features)[0]
    # Convert image to base64 encoding
    encoded_image = base64.b64encode(cv2.imencode('.jpg', cv2.imdecode(img_array, cv2.IMREAD_COLOR))[1]).decode('utf-8')
    # Render the same page with the prediction value
    return render_template('home.html', prediction=True, predicted_value = predicted_class, img_base64=encoded_image) 


if __name__ == "__main__":
    app.run(debug=True)

