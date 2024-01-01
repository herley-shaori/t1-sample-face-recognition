import cv2
import os

cascade_path = "haarcascade_frontalface_default.xml"  # Path to face cascade file
face_cascade = cv2.CascadeClassifier(cascade_path)

train_folder = "raw"
bounding_box_folder = "bounding_box_training_images"
cropped_images_folder = "cropped_images"

os.makedirs(bounding_box_folder, exist_ok=True)
os.makedirs(cropped_images_folder, exist_ok=True)

for filename in os.listdir(train_folder):
    img = cv2.imread(os.path.join(train_folder, filename))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)  # Blue bounding box
        cv2.imwrite(os.path.join(bounding_box_folder, filename), img)
        cropped_img = img[y:y+h, x:x+w]
        cv2.imwrite(os.path.join(cropped_images_folder, filename), cropped_img)
