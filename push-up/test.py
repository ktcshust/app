import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
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

def predict_label(landmarks, model):
    if landmarks is not None:
        landmarks = landmarks.reshape(1, 1, -1)
        label = model.predict(landmarks)[0][0]
        return label
    return None

# Load model
model = load_model('push-up/effnetrnn.h5')

# Đường dẫn đến video
video_path = "push-up/pushups-sample.mp4"

# Mở video
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Xử lý từng frame
    landmarks = extract_landmarks_from_frame(frame)
    label = predict_label(landmarks, model)
    
    # Vẽ biểu đồ lên frame
    if label is not None:
        cv2.putText(frame, f'Label: {label:.2f}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Hiển thị frame với biểu đồ
    cv2.imshow('Frame', frame)
    
    # Thoát nếu nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
