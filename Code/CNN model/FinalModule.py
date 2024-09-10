import os
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
import matplotlib.pyplot as plt
from google.colab import drive

# Kết nối Google Drive
drive.mount('/content/drive')

# Đường dẫn đến thư mục chứa dữ liệu trên Google Drive
base_dir = '/content/drive/MyDrive/Counting Dataset/'

# Đường dẫn đến các thư mục con
train_X_dir = os.path.join(base_dir, 'Train/X_train')
train_Y_dir = os.path.join(base_dir, 'Train/Y_train')

# Hàm để tải ảnh từ thư mục
def load_images_from_folder(folder):
    images = []
    for subdir in os.listdir(folder):
        subfolder = os.path.join(folder, subdir)
        for filename in os.listdir(subfolder):
            img_path = os.path.join(subfolder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
    return np.array(images)

# Tải ảnh từ các thư mục
X_data = load_images_from_folder(train_X_dir)
Y_data = load_images_from_folder(train_Y_dir)

# Thêm chiều cho dữ liệu (để có dạng [số lượng mẫu, chiều cao, chiều rộng, số kênh])
X_data = np.expand_dims(X_data, axis=-1)  # Thêm chiều kênh cho X_data
Y_data = np.expand_dims(Y_data, axis=-1)  # Thêm chiều kênh cho Y_data

# Chia dữ liệu thành 80% train và 20% validation
X_train, X_val, Y_train, Y_val = train_test_split(X_data, Y_data, test_size=0.2, random_state=42)

# Khởi tạo mô hình U-Net
inputs = Input(shape=(64, 64, 1))

# Module 1 (Block1)
conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
drop1 = Dropout(0.5)(conv2)
pool1 = MaxPooling2D(pool_size=(2, 2))(drop1)

# Module 2 (Block2)
conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
drop2 = Dropout(0.5)(conv4)
pool2 = MaxPooling2D(pool_size=(2, 2))(drop2)

# Module 3 (Block3)
conv5 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv5)
drop3 = Dropout(0.5)(conv6)
pool3 = MaxPooling2D(pool_size=(2, 2))(drop3)

# Module 4 (Block4)
conv7 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)
conv8 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv7)
up1 = UpSampling2D(size=(2, 2))(conv8)
merge1 = concatenate([conv6, up1], axis=3)

# Module 5 (Block5)
conv9 = Conv2D(256, (3, 3), activation='relu', padding='same')(merge1)
conv10 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv9)
up2 = UpSampling2D(size=(2, 2))(conv10)
merge2 = concatenate([conv4, up2], axis=3)

# Module 6 (Block6)
conv11 = Conv2D(128, (3, 3), activation='relu', padding='same')(merge2)
conv12 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv11)
up3 = UpSampling2D(size=(2, 2))(conv12)
merge3 = concatenate([conv2, up3], axis=3)

# Module 7 (Block7)
conv13 = Conv2D(64, (3, 3), activation='relu', padding='same')(merge3)
conv14 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv13)
conv15 = Conv2D(2, (3, 3), activation='relu', padding='same')(conv14)
conv16 = Conv2D(1, (1, 1), activation='sigmoid')(conv15)

# Hoàn thiện mô hình
model = Model(inputs=inputs, outputs=conv16)

# Biên dịch mô hình với hàm mất mát và trình tối ưu
model.compile(optimizer=Adam(learning_rate=1e-4), loss=binary_crossentropy, metrics=['accuracy'])

# Tạo các ImageDataGenerator để tăng cường dữ liệu
datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, rotation_range=90)

# Tạo generator cho dữ liệu huấn luyện
train_generator = datagen.flow(X_train, Y_train, batch_size=32)

# Callbacks để lưu mô hình tốt nhất và dừng sớm nếu không cải thiện
checkpoint = ModelCheckpoint('/content/drive/MyDrive/Counting Dataset/best_model.h5', monitor='val_loss', save_best_only=True, mode='min')
early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min')

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
