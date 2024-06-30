import cv2
from tensorflow.keras.models import load_model
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=2)
effnet_model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')

# Load the trained model
model = load_model('cnn.h5')

def process_frame(frame, pose, model, effnet_model):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()

        # Extract EfficientNet features
        img_data = cv2.resize(frame, (224, 224))
        img_data = img_data.astype(np.float32)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)
        effnet_features = effnet_model.predict(img_data)
        effnet_features = effnet_features.flatten()

        # Combine landmarks and EfficientNet features
        combined_features = np.concatenate((landmarks, effnet_features))
        combined_features = combined_features.reshape((1, -1, 1))
        
        # Predict the label using the CNN model
        prediction = model.predict(combined_features)
        return prediction[0][0]  # Return the predicted label (probability)
    return None

def create_plot_image(predictions, frame_count):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(predictions, label='Frame-wise Predictions')
    ax.set_xlim(0, frame_count)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Frame Number')
    ax.set_ylabel('Prediction (Probability)')
    ax.set_title('Frame-wise Predictions using Mediapipe+Effnet+CNN')
    ax.legend()
    canvas = FigureCanvas(fig)
    canvas.draw()
    img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    img = img.reshape(canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return img

def process_video(video_path, output_video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    frame_predictions = []
    frame_number = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        prediction = process_frame(frame, pose, model, effnet_model)
        frame_predictions.append(prediction)

        if prediction is not None:
            label = "Up" if prediction < 0.5 else "Down"
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Create the plot image for the current frame predictions
        plot_img = create_plot_image(frame_predictions, frame_number + 1)
        plot_img = cv2.resize(plot_img, (width // 2, height // 4))

        # Overlay the plot image onto the frame
        frame[-plot_img.shape[0]:, :plot_img.shape[1]] = plot_img
        
        out.write(frame)
        frame_number += 1

    cap.release()
    out.release()

    return frame_predictions

# Process the input video and get frame-wise predictions
video_path = 'pushups-sample.mp4'
output_video_path = 'output_video.mp4'
frame_predictions = process_video(video_path, output_video_path)
