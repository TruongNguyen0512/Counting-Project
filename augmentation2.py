import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import random

# Step 1: Load images
def load_images(directory_or_file):
    images = []
    if os.path.isdir(directory_or_file):
        for filename in os.listdir(directory_or_file):
            img_path = os.path.join(directory_or_file, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
    elif os.path.isfile(directory_or_file):
        img = cv2.imread(directory_or_file, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
    return images

# Step 2: Label images (Placeholder for labeling function)
def label_images(images):
    # Placeholder: Assuming images are already labeled
    return images

# Step 3: Cut images into 64x64 portions
def cut_image(image, size=(64, 64)):
    h, w = image.shape
    patches = []
    for i in range(0, h, size[0]):
        for j in range(0, w, size[1]):
            patch = image[i:i + size[0], j:j + size[1]]
            if patch.shape == size:
                patches.append(patch)
    return patches

# Step 4: Split dataset into training and validation sets
def split_dataset(images, validation_split=0.2):
    random.shuffle(images)
    split_idx = int(len(images) * validation_split)
    validation_set = images[:split_idx]
    training_set = images[split_idx:]
    return training_set, validation_set

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

# Step 5: Perform data augmentation
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

# Load original images
image_dir = 'image/test1.png'  # Directory containing the original images or single file
original_images = load_images(image_dir)

# Step 2: Label images (Placeholder)
labeled_images = label_images(original_images)

# Step 3: Cut images into 64x64 portions
all_patches = []
for img in labeled_images:
    patches = cut_image(img)
    all_patches.extend(patches)

# Step 4: Split dataset into training and validation sets
training_patches, validation_patches = split_dataset(all_patches)

# Step 5: Augment training data and save
output_dir = 'data/train'
os.makedirs(output_dir, exist_ok=True)

count = 0
for patch in training_patches:
    augmented_images = augment_image(patch)
    for aug_image in augmented_images:
        output_path = os.path.join(output_dir, f'augmented_{count}.png')
        cv2.imwrite(output_path, aug_image)
        count += 1

print(f"Generated {count} augmented images.")
