import os
import numpy as np
import cv2
import pandas as pd
import boto3
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

cropped_images_folder = 'cropped_images'
# Load cropped images and labels (assuming labels are in filenames)
X_train = []
y_train = []
for filename in os.listdir(cropped_images_folder):
    img = cv2.imread(os.path.join(cropped_images_folder, filename))
    X_train.append(img.flatten())  # Flatten for classification
    y_train.append(filename.split("_")[0])  # Assuming label is in filename

# Ensure consistent array shape
max_shape = (max([img.shape[0] for img in X_train]),)
X_train_padded = [np.pad(img, ((0, max_shape[0] - img.shape[0])), mode='constant') for img in X_train]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_train_padded, y_train, test_size=0.2, random_state=42)

# Create DataFrames from NumPy arrays:
X_train_df = pd.DataFrame(X_train)
X_test_df = pd.DataFrame(X_test)
y_train_df = pd.DataFrame(y_train)
y_test_df = pd.DataFrame(y_test)

# Automate feature names:
num_features = 56307
X_train_df.columns = [f'feature_{i}' for i in range(num_features)]
X_test_df.columns = X_train_df.columns  # Use the same column names for consistency

# Set target variable name:
y_train_df.columns = ['target']
y_test_df.columns = y_train_df.columns

# Add y_train as the first column of X_train_df:
X_train_df = pd.concat([y_train_df, X_train_df], axis=1)

X_train_df.to_csv('train.csv',index=False)


# Train SVM and Naive Bayes models
svm_model = SVC().fit(X_train, y_train)
nb_model = GaussianNB().fit(X_train, y_train)

# Evaluate models
currentF1Score = None
for model in [svm_model,nb_model]:
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    print("Model:", model.__class__.__name__)
    print("Confusion Matrix:\n", cm)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)
    # Save the best model.
    if(currentF1Score is None or f1 > currentF1Score):
        currentF1Score = f1
        with open(os.path.join('model.pkl'), 'wb') as out:
            pickle.dump(model, out)
        s3 = boto3.client('s3')
        s3.upload_file('model.pkl', 'temp-data-for-rekognition', 'byom-model/model.pkl')