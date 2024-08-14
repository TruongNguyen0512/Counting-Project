from google.colab import drive
import numpy as np
import os
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Kết nối với Google Drive
drive.mount('/content/drive')

# Đường dẫn đến thư mục chứa dữ liệu trên Google Drive
base_dir = '/content/drive/MyDrive/MyProjectDataset/'

# Đường dẫn đến các thư mục con
train_X_dir = os.path.join(base_dir, 'train/X_train')
train_Y_dir = os.path.join(base_dir, 'train/Y_train')
test_X_dir = os.path.join(base_dir, 'test/X_test')
test_Y_dir = os.path.join(base_dir, 'test/Y_test')

# Hàm để tải dữ liệu từ các tệp .npy
def load_data(data_dir):
    data_list = []
    for file_name in sorted(os.listdir(data_dir)):
        file_path = os.path.join(data_dir, file_name)
        data = np.load(file_path)
        data_list.append(data)
    return np.array(data_list)

# Tải dữ liệu từ Google Drive
X_train = load_data(train_X_dir)
Y_train = load_data(train_Y_dir)
X_test = load_data(test_X_dir)
Y_test = load_data(test_Y_dir)

# Kiểm tra kích thước dữ liệu đã tải
print(f"X_train shape: {X_train.shape}")
print(f"Y_train shape: {Y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"Y_test shape: {Y_test.shape}")

# Khởi tạo mô hình
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
model.compile(optimizer=Adam(lr=1e-4), loss=binary_crossentropy, metrics=['accuracy'])

# Tạo các ImageDataGenerator để tăng cường dữ liệu nếu cần
datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, rotation_range=90)

# Callbacks để lưu mô hình tốt nhất và dừng sớm nếu không cải thiện
checkpoint = ModelCheckpoint('/content/drive/MyDrive/MyProjectDataset/best_model.h5', monitor='val_loss', save_best_only=True, mode='min')
early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min')

# Huấn luyện mô hình
history = model.fit(datagen.flow(X_train, Y_train, batch_size=32),
                    validation_data=(X_test, Y_test),
                    epochs=50,
                    callbacks=[checkpoint, early_stopping])

# Đánh giá mô hình
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Vẽ biểu đồ quá trình huấn luyện
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='Training accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.legend()
plt.show()
