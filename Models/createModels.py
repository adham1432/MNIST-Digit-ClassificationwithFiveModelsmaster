# Import necessary libraries
import numpy as np
from keras.layers import Dense
from keras import Sequential
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from keras.models import save_model
import pickle
import os
import joblib

# -------------------------------------- preparing data -------------------------------------
# Load the input features
X_train_features = np.load('preparingData\X_train_features.npy')
X_test_features = np.load('preparingData\X_test_features.npy')
# Load the labels using pickle
with open('preparingData\\train_labels.pkl', 'rb') as f:
    train_labels = pickle.load(f)
with open('preparingData\\test_labels.pkl', 'rb') as f:
    test_labels = pickle.load(f)
# Show shapes of the train and test data
print(X_train_features.shape)
print(X_test_features.shape)
print(train_labels.shape)
print(test_labels.shape)

# ---------------------------------------------------------------------------------------------
# ------------------------------------- Random Forest -----------------------------------------
# ---------------------------------------------------------------------------------------------

# Create random forest class
class RFModel():
    def __init__(self, n = 150, crit = 'gini', min_ss = 2, min_sl = 1):
        self.model = RandomForestClassifier(n_estimators=n,    
                                    criterion = crit, 
                                    max_depth = 50,     
                                    min_samples_split = min_ss,  
                                    min_samples_leaf = min_sl,
                                    n_jobs=-1)  
    def fit(self, train_data, labels):
        # Create random forest model
        self.model.fit(train_data, labels)
    # Make predictions with the random forest model
    def predict(self, data):
        predictions = self.model.predict(data)
        return predictions
    def Report(self, pred_data, actual_data):
        # Generate and print the classification report
        report = classification_report(actual_data, pred_data)
        return report
    def save_model(self, filename):
        # Save model if not exist in the determined path
        if not os.path.exists(filename):
            # Save the model
            joblib.dump(self.model, filename)

# Create random forest model
RF_model = RFModel()
RF_model.fit(X_train_features, train_labels)
rf_predictions = RF_model.predict(X_test_features)
rf_report = RF_model.Report(rf_predictions, test_labels)
print("Decision Tree Classification Report:\n", rf_report)
RF_model.save_model('Models\\random_forest_model.joblib')

# ---------------------------------------------------------------------------------------------
# ------------------------------------- Decision Tree -----------------------------------------
# ---------------------------------------------------------------------------------------------
# Create Decision Tree class
class DTModel():
    def __init__(self,maxD = 50, min_ss = 10, min_sl = 5):
        # Create the Decision Tree model
        self.decision_tree = DecisionTreeClassifier(max_depth = maxD,     
                                                    min_samples_split = min_ss,  
                                                    min_samples_leaf = min_sl,)
    def fit(self, train_data, labels):
        # Train the Decision Tree model
        self.decision_tree.fit(train_data, labels)
    def predict(self, data):
        predictions = self.decision_tree.predict(data)
        return predictions
    def Report(self, pred_data, actual_data):
        # Generate and print the classification report
        report = classification_report(actual_data, pred_data)
        return report
    def save_model(self, filename):
        # Save model if not exist in the determined path
        if not os.path.exists(filename):
            # Save the model
            joblib.dump(self.decision_tree, filename)

# Create Decision Tree model
decision_tree = DTModel()
decision_tree.fit(X_train_features, train_labels)
dt_predictions = decision_tree.predict(X_test_features)
dt_report = decision_tree.Report(dt_predictions, test_labels)
print("Decision Tree Classification Report:\n", dt_report)
decision_tree.save_model('Models\decision_tree_model.joblib')

# ---------------------------------------------------------------------------------------------
# ------------------------------------- Deep Neural Network -----------------------------------
# ---------------------------------------------------------------------------------------------

# Create Deep Neural Network class
class DNNModel():
    def __init__(self):
        self.model = Sequential([Dense(256, activation='relu', 
                                        input_shape = (512,)),
                                Dense(128, activation='relu'),
                                Dense(64, activation='relu'),
                                Dense(10, activation='softmax')
                            ])
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    def fit(self, train_data, label_data):
        history = self.model.fit(train_data, label_data, epochs=10, validation_split=0.1)
        return history
    # Visualize training and validation results
    def visualize_result(self, result):
        plt.figure(figsize=(10, 5))
        # Visualize the accuracy
        plt.subplot(1, 2, 1)
        plt.plot(result.history['accuracy'], label='Training Accuracy')
        plt.plot(result.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        # Visualize the loss
        plt.subplot(1, 2, 2)
        plt.plot(result.history['loss'], label='Training Loss')
        plt.plot(result.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.tight_layout()
        plt.show()
    # Evaluate the model
    def evaluateModel(self, test_data, label_data):
        test_loss, test_acc = self.model.evaluate(test_data, label_data)
        print('Test Accuracy:', test_acc)
        print('Test Loss:', test_loss)
    # Adjust the predict function to return only the predicted labels
    def predict(self, test_data):
        predictions = self.model.predict(test_data)
        predicted_labels = np.argmax(predictions, axis=1)  # Convert softmax output to class labels
        return predicted_labels
    # Create Classification Report
    def report(self, actual_labels, predicted_labels):
        class_report = classification_report(actual_labels, predicted_labels)
        return class_report
    # Save the Deep neural network model if already not exist
    def save_model(self, filename):
        if not os.path.exists(filename):
            save_model(self.model, filename)

# Create Deep Neural Network model
DnnModel = DNNModel()
Dnnresult = DnnModel.fit(X_train_features, train_labels)
DnnModel.visualize_result(Dnnresult)
DnnModel.evaluateModel(X_test_features, test_labels)
DNNpredictions = DnnModel.predict(X_test_features)
DNNreport = DnnModel.report(test_labels, DNNpredictions)
print("DNN Classification Report:\n", DNNreport)
DnnModel.save_model('Models\dnn_model.h5')

# ---------------------------------------------------------------------------------------------
# ------------------------------- Naive bayesian classifier -----------------------------------
# ---------------------------------------------------------------------------------------------

# Create naive bayesian class
class NBclassifier:
    def __init__(self):
        self.model = GaussianNB()
    def fit(self, train_data, labels):
        self.model.fit(train_data, labels)
    # Make predictions
    def predict(self, test_data):
        predictions = self.model.predict(test_data)
        return predictions
    # Evaluate the model
    def evaluate(self, actual_data, predicted_data):
        accuracy = accuracy_score(actual_data, predicted_data)
        report = classification_report(actual_data, predicted_data)
        return accuracy, report
    def save_model(self, filename):
        # Save model if not exist in the determined path
        if not os.path.exists(filename):
            # Save the model
            joblib.dump(self.model, filename)

# Create naive bayesian model
NBmodel = NBclassifier()
# Train the model
NBmodel.fit(X_train_features, train_labels)
# Make predictions
BC_predictions = NBmodel.predict(X_test_features)
# Evaluate the model
NBacc, NBreport = NBmodel.evaluate(test_labels, BC_predictions)
print("the model accuracy is", NBacc)
print("the classification report is\n",NBreport)
NBmodel.save_model('Models\\naiveclassifier.joblib')

# ---------------------------------------------------------------------------------------------
# ------------------------------- K Nearest Neighbors Classifier ------------------------------
# ---------------------------------------------------------------------------------------------

# Create naive bayesian class
class KNClassifier:
    def __init__(self, n = 5):
        self.model = KNeighborsClassifier(n_neighbors = n)
    def fit(self, train_data, labels):
        self.model.fit(train_data, labels)
    def predict(self, data):
        predictions = self.model.predict(data)
        return predictions
    def acc_report(self, actual_data, predicted_data):
        # Calculate accuracy and report
        accuracy = accuracy_score(actual_data, predicted_data)
        report = classification_report(test_labels, knn_predictions)
        return accuracy, report
    def save_model(self, filename):
        # Save model if not exist in the determined path
        if not os.path.exists(filename):
            # Save the model
            joblib.dump(self.model, filename)

# Create and fit the KNN model
knn = KNClassifier()
knn.fit(X_train_features, train_labels)
# Make predictions on the test data
knn_predictions = knn.predict(X_test_features)
# Find accuracy and classification report 
knn_acc, knn_report = knn.acc_report(test_labels, knn_predictions)
# Generate accuracy and classification report
print("KNN accuracy:\n", knn_acc)
print("KNN Classification Report:\n", knn_report)
# Save model
knn.save_model('Models\knn.joblib')



