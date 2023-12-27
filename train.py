import joblib
import os
import numpy as np
import cv2
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
    print(len(img.flatten()))
    X_train.append(img.flatten())  # Flatten for classification
    y_train.append(filename.split("_")[0])  # Assuming label is in filename

# Ensure consistent array shape
max_shape = (max([img.shape[0] for img in X_train]),)
X_train_padded = [np.pad(img, ((0, max_shape[0] - img.shape[0])), mode='constant') for img in X_train]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_train_padded, y_train, test_size=0.2, random_state=42)

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
        joblib.dump(model, "model.pkl")
    # if(f1_score is None or f1 > f1_score):
    #     f1_score = f1
    #     best_model = model
    #     joblib.dump(best_model, "model.pkl")