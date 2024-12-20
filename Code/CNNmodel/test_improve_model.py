# Code/CNNmodel/FinalModule.py

import os
import tensorflow
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Đường dẫn đến các thư mục con
train_X_dir = '../../data/train/x_train'
train_Y_dir = '../../data/train/y_train'

# Hàm để tải ảnh từ thư mục
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return np.array(images)

# Tải ảnh từ các thư mục
X_data = load_images_from_folder(train_X_dir)
Y_data = load_images_from_folder(train_Y_dir)

# Normalize the data
X_data = X_data / 255.0
Y_data = Y_data / 255.0

# Thêm chiều cho dữ liệu (để có dạng [số lượng mẫu, chiều cao, chiều rộng, số kênh])
X_data = np.expand_dims(X_data, axis=-1)  # Thêm chiều kênh cho X_data
Y_data = np.expand_dims(Y_data, axis=-1)  # Thêm chiều kênh cho Y_data

print(f"Number of X_data samples: {len(X_data)}")
print(f"Number of Y_data samples: {len(Y_data)}")

# Chia dữ liệu thành 80% train và 20% validation
X_train, X_val, Y_train, Y_val = train_test_split(X_data, Y_data, test_size=0.2, random_state=42)

# Khởi tạo mô hình U-Net
inputs = Input(shape=(64, 64, 1))

# Module 1 (Block1)
conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
conv1 = BatchNormalization()(conv1)
conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
conv2 = BatchNormalization()(conv2)
drop1 = Dropout(0.5)(conv2)
pool1 = MaxPooling2D(pool_size=(2, 2))(drop1)

# Module 2 (Block2)
conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
conv3 = BatchNormalization()(conv3)
conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
conv4 = BatchNormalization()(conv4)
drop2 = Dropout(0.5)(conv4)
pool2 = MaxPooling2D(pool_size=(2, 2))(drop2)

# Module 3 (Block3)
conv5 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
conv5 = BatchNormalization()(conv5)
conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv5)
conv6 = BatchNormalization()(conv6)
drop3 = Dropout(0.5)(conv6)
pool3 = MaxPooling2D(pool_size=(2, 2))(drop3)

# Module 4 (Block4)
conv7 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)
conv7 = BatchNormalization()(conv7)
conv8 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv7)
conv8 = BatchNormalization()(conv8)
drop4 = Dropout(0.5)(conv8)
pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

# Module 5 (Block5)
conv9 = Conv2D(1024, (3, 3), activation='relu', padding='same')(pool4)
conv9 = BatchNormalization()(conv9)
conv10 = Conv2D(1024, (3, 3), activation='relu', padding='same')(conv9)
conv10 = BatchNormalization()(conv10)
drop5 = Dropout(0.5)(conv10)

# Upsampling
up1 = UpSampling2D(size=(2, 2))(drop5)
merge1 = concatenate([conv8,up1], axis=3)
conv11 = Conv2D(512, (3, 3), activation='relu', padding='same')(merge1)
conv11 = BatchNormalization()(conv11)
conv12 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv11)
conv12 = BatchNormalization()(conv12)

up2 = UpSampling2D(size=(2, 2))(conv12)
merge2 = concatenate([conv6, up2], axis=3)
conv13 = Conv2D(256, (3, 3), activation='relu', padding='same')(merge2)
conv13 = BatchNormalization()(conv13)
conv14 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv13)
conv14 = BatchNormalization()(conv14)

up3 = UpSampling2D(size=(2, 2))(conv14)
merge3 = concatenate([conv4, up3], axis=3)
conv15 = Conv2D(128, (3, 3), activation='relu', padding='same')(merge3)
conv15 = BatchNormalization()(conv15)
conv16 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv15)
conv16 = BatchNormalization()(conv16)

up4 = UpSampling2D(size=(2, 2))(conv16)
merge4 = concatenate([conv2, up4], axis=3)
conv17 = Conv2D(64, (3, 3), activation='relu', padding='same')(merge4)
conv17 = BatchNormalization()(conv17)
conv18 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv17)
conv18 = BatchNormalization()(conv18)

conv19 = Conv2D(1, (1, 1), activation='sigmoid')(conv18)

# Hoàn thiện mô hình
model = Model(inputs=inputs, outputs=conv19)

# Biên dịch mô hình với hàm mất mát và trình tối ưu
model.compile(optimizer=Adam(learning_rate=1e-4, clipnorm=1.0), loss=MeanSquaredError(), metrics=['accuracy'])

# Tạo các ImageDataGenerator để tăng cường dữ liệu
datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, rotation_range=90)

# Tạo generator cho dữ liệu huấn luyện
train_generator = datagen.flow(X_train, Y_train, batch_size=32)

# Callbacks để lưu mô hình tốt nhất và dừng sớm nếu không cải thiện
checkpoint = ModelCheckpoint('../../best_model.keras', monitor='val_loss', save_best_only=True, mode='min')
early_stopping = EarlyStopping(monitor='val_loss', patience=20, mode='min', restore_best_weights=True)

# Huấn luyện mô hình với dữ liệu tăng cường
history = model.fit(train_generator,
                    validation_data=(X_val, Y_val),
                    epochs=50,
                    callbacks=[checkpoint, early_stopping])

# Đánh giá mô hình
score = model.evaluate(X_val, Y_val, verbose=0)
print('Validation loss:', score[0])
print('Validation accuracy:', score[1])

# Vẽ biểu đồ quá trình huấn luyện
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='Training accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.legend()
plt.show()