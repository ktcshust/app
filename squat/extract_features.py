import cv2
import mediapipe as mp
import os
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=2)

def extract_landmarks(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read image at {image_path}")
        return None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
        return landmarks
    else:
        print(f"Warning: No pose landmarks detected in image {image_path}")
    return None

def augment_data(image_path, output_folder):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read image at {image_path}")
        return
    
    # Example augmentation: flip horizontally
    augmented_image = cv2.flip(image, 1)  # 1 for horizontal flip
    
    # Save augmented image
    filename = os.path.basename(image_path)
    output_path = os.path.join(output_folder, f"augmented_{filename}")
    cv2.imwrite(output_path, augmented_image)

    return output_path

def augment_images_in_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    augmented_image_paths = []
    for filename in os.listdir(input_folder):
        image_path = os.path.join(input_folder, filename)
        augmented_path = augment_data(image_path, output_folder)
        augmented_image_paths.append(augmented_path)
    
    return augmented_image_paths

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input

# Load pre-trained ResNet50 model and remove the top layer
resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

def extract_resnet_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    resnet_features = resnet_model.predict(img_data)
    return resnet_features.flatten()


# Load pre-trained EfficientNetB0 model and remove the top layer
effnet_model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')

def extract_effnet_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    effnet_features = effnet_model.predict(img_data)
    return effnet_features.flatten()

def load_data_from_paths(image_paths, label):
    data = []
    data1 = []
    data2 = []
    labels = []
    for image_path in image_paths:
        landmarks = extract_landmarks(image_path)
        if landmarks is not None:
            resnet_features = extract_resnet_features(image_path)
            effnet_features = extract_effnet_features(image_path)
            combined_features1 = np.concatenate((landmarks, resnet_features))
            combined_features2 = np.concatenate((landmarks, effnet_features))
            data.append(landmarks)
            data1.append(combined_features1)
            data2.append(combined_features2)
            labels.append(label)
    return data, data1, data2, labels

def load_data(folder, label):
    data = []
    data1 = []
    data2 = []
    labels = []
    for filename in os.listdir(folder):
        image_path = os.path.join(folder, filename)
        landmarks = extract_landmarks(image_path)
        if landmarks is not None:
            resnet_features = extract_resnet_features(image_path)
            effnet_features = extract_effnet_features(image_path)
            combined_features1 = np.concatenate((landmarks, resnet_features))
            combined_features2 = np.concatenate((landmarks, effnet_features))
            data.append(landmarks)
            data1.append(combined_features1)
            data2.append(combined_features2)
            labels.append(label)
    return data, data1, data2, labels

# Augment and load data
down_augmented_paths = augment_images_in_folder("squat/down", "squat/down_augmented")
up_augmented_paths = augment_images_in_folder("squat/up", "squat/up_augmented")

down_data, down_data_resnet, down_data_effnet, down_labels = load_data("squat/down", 1)
up_data, up_data_resnet, up_data_effnet, up_labels = load_data("squat/up", 0)

down_data1, down_data_resnet1, down_data_effnet1, down_labels1 = load_data_from_paths(down_augmented_paths, 1)
up_data1, up_data_resnet1, up_data_effnet1, up_labels1 = load_data_from_paths(up_augmented_paths, 0)

X = np.array(down_data + down_data1 + up_data + up_data1)
y = np.array(down_labels + down_labels1 + up_labels + up_labels1)

X_resnet = np.array(down_data_resnet + down_data_resnet1 + up_data_resnet + up_data_resnet1)

X_effnet = np.array(down_data_effnet + down_data_effnet1 + up_data_effnet + up_data_effnet1)

np.save('squat/X.npy', X)
np.save('squat/y.npy', y)
np.save('squat/X1.npy', X_resnet)
np.save('squat/X2.npy', X_effnet)
print("Saved X.npy and y.npy.")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"Sample label y[0]: {y[0]}")
print(f"Sample label y[-1]: {y[-1]}")
print(f"Total number of samples: {len(y)}")


