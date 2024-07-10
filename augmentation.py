import cv2
import numpy as np
import os
from glob import glob

# Function to cut the image into 64x64 portions
def cut_image(img, cut_size=(64, 64)):
    img_height, img_width = img.shape
    cuts = []
    for y in range(0, img_height, cut_size[1]):
        for x in range(0, img_width, cut_size[0]):
            cut = img[y:y + cut_size[1], x:x + cut_size[0]]
            if cut.shape == cut_size:
                cuts.append(cut)
    return cuts

# Augmentation functions
def blur_image(img):
    return cv2.GaussianBlur(img, (5, 5), 0)

def scale_image(img, scale=0.9):
    height, width = img.shape
    scaled_img = cv2.resize(img, (int(width*scale), int(height*scale)))
    return cv2.resize(scaled_img, (width, height))

def rotate_image(img, angle=15):
    height, width = img.shape
    M = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    return cv2.warpAffine(img, M, (width, height))

def change_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    hsv = cv2.cvtColor(hsv, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, value)
    v[v > 255] = 255
    v[v < 0] = 0
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def add_noise(img):
    noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
    noisy_img = cv2.add(img, noise)
    return noisy_img

# Load the original image
original_image_path = 'image/test1.png'
img = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)

# Cut the image into 64x64 portions
cuts = cut_image(img)

# Augment each cut portion
augmented_images = []
for cut in cuts:
    augmented_images.append(cut)
    augmented_images.append(blur_image(cut))
    augmented_images.append(scale_image(cut))
    augmented_images.append(rotate_image(cut))
    augmented_images.append(change_brightness(cut))
    augmented_images.append(add_noise(cut))

# Save the augmented images
output_dir = 'data/train'
os.makedirs(output_dir, exist_ok=True)
for i, aug_img in enumerate(augmented_images):
    cv2.imwrite(os.path.join(output_dir, f'augmented_{i}.png'), aug_img)

print(f'Generated {len(augmented_images)} augmented images.')
