import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from tkinter import Listbox
import cv2
from tensorflow.keras.models import load_model
import numpy as np
import mediapipe as mp
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image, ImageTk
import os
import tensorflow as tf
import tf2onnx
import onnxruntime as ort

history = []


mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=2)
effnet_model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')


# Define the output path for the ONNX model
output_path = "efficientnet_b0.onnx"

# Convert the TensorFlow model to ONNX format
spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
model_proto, _ = tf2onnx.convert.from_keras(effnet_model, input_signature=spec, opset=13)
with open(output_path, "wb") as f:
    f.write(model_proto.SerializeToString())


model = load_model('push-up/cnn.h5')

# Load the ONNX model
onnx_model_path = 'efficientnet_b0.onnx'
ort_session = ort.InferenceSession(onnx_model_path)

def extract_effnet_features_onnx(img_data):
    # Preprocess the input image
    img_data = preprocess_input(img_data)

    # Run the ONNX model
    ort_inputs = {ort_session.get_inputs()[0].name: img_data}
    ort_outs = ort_session.run(None, ort_inputs)

    return ort_outs[0].flatten()

def process_frame(frame, pose, model, ort_session):
    frame = cv2.resize(frame, (480, 270))
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()

        # Extract EfficientNet features using ONNX model
        img_data = cv2.resize(frame, (224, 224))
        img_data = img_data.astype(np.float32)
        img_data = np.expand_dims(img_data, axis=0)
        effnet_features = extract_effnet_features_onnx(img_data)

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

def process_video(video_path, canvas, history_listbox, progress_bar):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return

    frame_predictions = []
    frame_number = 0
    frame_interval = 5  # Only process every 5th frame

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(f'processed_{os.path.basename(video_path)}', fourcc, cap.get(cv2.CAP_PROP_FPS) / frame_interval, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_processed_frames = total_frames // frame_interval

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_number % frame_interval == 0:
            prediction = process_frame(frame, pose, model, effnet_model)
            frame_predictions.append(prediction)

            if prediction is not None:
                label = "Up" if prediction < 0.5 else "Down"
                cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            plot_img = create_plot_image(frame_predictions, frame_number // frame_interval + 1)
            plot_img = cv2.resize(plot_img, (frame.shape[1] // 2, frame.shape[0] // 4))

            frame[-plot_img.shape[0]:, :plot_img.shape[1]] = plot_img

            out.write(frame)
            # Update progress bar
            progress_bar['value'] = (frame_number // frame_interval) / total_processed_frames * 100
            app.update_idletasks()

        frame_number += 1

    cap.release()
    out.release()

    processed_video_path = f'processed_{os.path.basename(video_path)}'
    save_to_history(processed_video_path, history_listbox)
    return processed_video_path

def select_video():
    file_path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4")])
    if file_path:
        processed_video_path = process_video(file_path, canvas, history_listbox, progress_bar)
        play_video(processed_video_path, canvas)

def save_to_history(file_path, history_listbox):
    history.append(file_path)
    history_listbox.insert(tk.END, os.path.basename(file_path))

def play_from_history(event):
    selected_index = history_listbox.curselection()
    if selected_index:
        video_path = history[selected_index[0]]
        play_video(video_path, history_canvas)

def play_video(video_path, canvas):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return

    def update_frame():
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return

        # Resize frame for display
        frame_resized = cv2.resize(frame, (canvas.winfo_width(), canvas.winfo_height()))
        img = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)

        canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        canvas.image = imgtk

        # Update frame at intervals
        canvas.after(int(1000 // cap.get(cv2.CAP_PROP_FPS)), update_frame)

    # Clear canvas and release previous video capture if any
    canvas.delete("all")
    cap.release()

    cap = cv2.VideoCapture(video_path)
    update_frame()

def select_exercise():
    exercise = exercise_var.get()
    if exercise == "push-up":
        model_path = "push-up/cnn.h5"
    elif exercise == "pull-up":
        model_path = "pull-up/cnn.h5"
    elif exercise == "squat":
        model_path = "squat/cnn.h5"
    else:
        messagebox.showwarning("Exercise Error", "Please select a valid exercise.")
        return

    global model
    model = load_model(model_path)
    select_video()

app = tk.Tk()
app.title("Exercise Video Processor")

exercise_var = tk.StringVar()
exercise_var.set("push-up")

tab_control = ttk.Notebook(app)

process_tab = ttk.Frame(tab_control)
history_tab = ttk.Frame(tab_control)

tab_control.add(process_tab, text='Process Video')
tab_control.add(history_tab, text='History')

tab_control.pack(expand=1, fill='both')

# Process Video Tab
tk.Label(process_tab, text="Select Exercise:").pack()

tk.Radiobutton(process_tab, text="Push-up", variable=exercise_var, value="push-up").pack()
tk.Radiobutton(process_tab, text="Pull-up", variable=exercise_var, value="pull-up").pack()
tk.Radiobutton(process_tab, text="Squat", variable=exercise_var, value="squat").pack()

tk.Button(process_tab, text="Select Video and Process", command=select_exercise).pack()

canvas = tk.Canvas(process_tab, width=640, height=480)
canvas.pack()

# Add Progressbar
progress_bar = ttk.Progressbar(process_tab, orient='horizontal', length=400, mode='determinate')
progress_bar.pack(pady=10)

# History Tab
history_canvas = tk.Canvas(history_tab, width=640, height=480)
history_canvas.pack()

tk.Label(history_tab, text="History:").pack()

history_listbox = Listbox(history_tab)
history_listbox.pack()
history_listbox.bind('<<ListboxSelect>>', play_from_history)

app.mainloop()



