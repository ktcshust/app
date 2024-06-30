import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import tensorflow
from tensorflow.keras.models import load_model

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=2)

def extract_landmarks_from_frame(frame):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
        return landmarks
    return None

def predict_video_labels(video_path, model):
    cap = cv2.VideoCapture(video_path)
    labels = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        landmarks = extract_landmarks_from_frame(frame)
        if landmarks is not None:
            landmarks = landmarks.reshape(1, 1, -1)
            label = model.predict(landmarks)
            labels.append(label[0][0])
    cap.release()
    return labels

# Load mô hình
model = load_model('pushup_rnn_model.h5')

# Đường dẫn đến video
video_path = "pushups-sample.mp4"

# Dự đoán nhãn cho từng frame
labels = predict_video_labels(video_path, model)

# Vẽ biểu đồ
plt.plot(labels)
plt.xlabel('Frame')
plt.ylabel('Label')
plt.title('Push Up Prediction')
plt.show()

