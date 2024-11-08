from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate
from keras.models import Model

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

# Hiển thị cấu trúc mô hình
model.summary()
