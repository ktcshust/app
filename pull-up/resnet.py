import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.applications import ResNet50
from keras.models import Model
from keras.layers import Dense, Flatten, Input
from keras.optimizers import Adam

# Load dữ liệu
X = np.load('X.npy')
y = np.load('y.npy')

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape dữ liệu cho mô hình ResNet
# Giả sử rằng dữ liệu X có kích thước (số mẫu, chiều cao, chiều rộng, số kênh)
# Nếu dữ liệu của bạn chỉ có 2D, bạn có thể cần thêm 1 chiều kênh
if len(X_train.shape) == 3:
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))

# Xây dựng mô hình ResNet
input_tensor = Input(shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]))
base_model = ResNet50(include_top=False, weights=None, input_tensor=input_tensor)

# Thêm các lớp fully connected vào cuối mô hình ResNet
x = Flatten()(base_model.output)
x = Dense(64, activation='relu')(x)
output_tensor = Dense(1, activation='sigmoid')(x)

# Kết hợp mô hình
model = Model(inputs=base_model.input, outputs=output_tensor)

model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy', "Precision", "Recall"])

# Huấn luyện mô hình
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Lưu mô hình
model.save('pushup_resnet_model.h5')

