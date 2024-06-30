import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
from keras.optimizers import Adam

# Load dữ liệu
X = np.load('squat/X.npy')
y = np.load('squat/y.npy')

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape dữ liệu cho mô hình CNN (Conv1D)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Xây dựng mô hình CNN
model = Sequential()
model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(64, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy', 'Precision', 'Recall'])

import matplotlib.pyplot as plt
# Huấn luyện mô hình
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Vẽ biểu đồ biểu diễn tất cả các val metrics và val loss qua từng epoch
epochs = range(1, len(history.history['val_loss']) + 1)

plt.figure(figsize=(12, 8))
plt.plot(epochs, history.history['val_loss'], label='Validation Loss')
plt.plot(epochs, history.history['val_accuracy'], label='Validation Accuracy')
plt.plot(epochs, history.history['val_Precision'], label='Validation Precision')
plt.plot(epochs, history.history['val_Recall'], label='Validation Recall')

plt.xlabel('Epochs')
plt.ylabel('Metrics')
plt.legend()
plt.title('Validation Metrics and Loss over Epochs in Mediapipe+CNN')
plt.show()

plt.savefig('squat/mediapipe_cnn.png')


model.save('squat/cnn1.h5')

