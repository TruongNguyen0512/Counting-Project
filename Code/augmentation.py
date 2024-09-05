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
            # Kiểm tra xem phần cắt có kích thước 64x64 không
            if cut.shape[0] == cut_size[1] and cut.shape[1] == cut_size[0]:
                cuts.append(cut)
    return cuts

# Augmentation functions
def blur_image(img):
    return cv2.GaussianBlur(img, (5, 5), 0)

def scale_image(img, scale=0.9):
    height, width = img.shape
    scaled_img = cv2.resize(img, (int(width * scale), int(height * scale)))
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

# Function to process images and their corresponding labels
def process_images_and_labels(input_img_folder, input_label_folder, output_img_folder, output_label_folder):
    # List of augmentation types
    augmentation_types = ['original', 'blurred', 'scaled', 'rotated', 'brightened', 'noisy']

    # Initialize counters for each augmentation type
    counts = {aug: 0 for aug in augmentation_types}

    # Ensure output folders exist for both images and labels
    for aug in augmentation_types:
        os.makedirs(os.path.join(output_img_folder, aug), exist_ok=True)
        os.makedirs(os.path.join(output_label_folder, aug), exist_ok=True)

    # Get list of all image files in the input folder
    image_paths = sorted(glob(os.path.join(input_img_folder, '*.png')))
    label_paths = sorted(glob(os.path.join(input_label_folder, '*.png')))

    if not image_paths or not label_paths:
        print('No images or labels found in the input folders')
        return

    for img_index, (img_path, lbl_path) in enumerate(zip(image_paths, label_paths)):
        # Read the image and label
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(lbl_path, cv2.IMREAD_GRAYSCALE)

        # Process the image and label
        img_cuts = cut_image(img)
        label_cuts = cut_image(label)

        if not img_cuts or not label_cuts or len(img_cuts) != len(label_cuts):
            continue

        # Process and save the images and labels
        for i, (img_cut, label_cut) in enumerate(zip(img_cuts, label_cuts)):
            # Apply augmentations to both image and label
            augmented_images = {
                'original': img_cut,
                'blurred': blur_image(img_cut),
                'scaled': scale_image(img_cut),
                'rotated': rotate_image(img_cut),
                'brightened': change_brightness(img_cut),
                'noisy': add_noise(img_cut)
            }
            augmented_labels = {
                'original': label_cut,
                'blurred': blur_image(label_cut),
                'scaled': scale_image(label_cut),
                'rotated': rotate_image(label_cut),
                'brightened': label_cut,  # Không cần thay đổi độ sáng nhãn
                'noisy': label_cut  # Không cần thêm nhiễu nhãn
            }

            for aug_type in augmentation_types:
                img_filename = f'image_{img_index}_{i}_{aug_type}.png'
                label_filename = f'label_{img_index}_{i}_{aug_type}.png'
                cv2.imwrite(os.path.join(output_img_folder, aug_type, img_filename), augmented_images[aug_type])
                cv2.imwrite(os.path.join(output_label_folder, aug_type, label_filename), augmented_labels[aug_type])
                counts[aug_type] += 1

    # Print the counts for each augmentation type
    for aug in augmentation_types:
        print(f'Số lượng ảnh trong thư mục {aug}: {counts[aug]}')

# Example usage
input_img_folder = 'image/united image gray'
input_label_folder = 'image/united image labeled'
output_img_folder = 'data/train/x_train'
output_label_folder = 'data/train/y_train'
process_images_and_labels(input_img_folder, input_label_folder, output_img_folder, output_label_folder)
