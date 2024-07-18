import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import random

# Function to load the image
def load_image(image_path):
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Function to cut the image into 64x64 portions
def cut_image(image, size=(64, 64)):
    h, w = image.shape
    patches = []
    for i in range(0, h, size[0]):
        for j in range(0, w, size[1]):
            patch = image[i:i + size[0], j:j + size[1]]
            if patch.shape == size:
                patches.append(patch)
    return patches

# Data augmentation functions
def blur_image(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

def scale_image(image, scale_range=(0.9, 1.1)):
    scale = random.uniform(scale_range[0], scale_range[1])
    h, w = image.shape
    resized = cv2.resize(image, (int(w * scale), int(h * scale)))
    if scale > 1:
        center = resized.shape[0] // 2, resized.shape[1] // 2
        cropped = resized[center[0] - h // 2:center[0] + h // 2, center[1] - w // 2:center[1] + w // 2]
    else:
        padded = cv2.copyMakeBorder(resized, 0, h - resized.shape[0], 0, w - resized.shape[1], cv2.BORDER_CONSTANT, value=[0, 0, 0])
        cropped = padded
    return cropped

def rotate_image(image, angle_range=(-10, 10)):
    angle = random.uniform(angle_range[0], angle_range[1])
    h, w = image.shape
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    rotated = cv2.warpAffine(image, matrix, (w, h))
    return rotated

def change_brightness(image, brightness_range=(0.5, 1.5)):
    enhancer = ImageEnhance.Brightness(Image.fromarray(image))
    factor = random.uniform(brightness_range[0], brightness_range[1])
    brightened = enhancer.enhance(factor)
    return np.array(brightened)

def add_noise(image, noise_level=0.1):
    noise = np.random.normal(0, noise_level, image.shape)
    noisy_image = image + noise * 255
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image

# Function to perform all augmentations
def augment_image(image):
    augmentations = [
        blur_image,
        scale_image,
        rotate_image,
        change_brightness,
        add_noise
    ]
    augmented_images = []
    for func in augmentations:
        augmented_images.append(func(image))
    return augmented_images

# Load the original image
image_path = 'image/test4.png'
image = load_image(image_path)

# Cut the image into 64x64 portions
patches = cut_image(image)

# Perform data augmentation and save the new dataset
output_dir = 'data/train'
os.makedirs(output_dir, exist_ok=True)

count = 0
for patch in patches:
    augmented_images = augment_image(patch)
    for aug_image in augmented_images:
        output_path = os.path.join(output_dir, f'augmented_{count}.png')
        cv2.imwrite(output_path, aug_image)
        count += 1

print(f"Generated {count} augmented images.")
