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

# Function to count images in a directory
def count_images_in_folder(folder):
    return len(glob(os.path.join(folder, '*.png')))

# Function to process all images in a folder
def process_images_in_folder(input_folder, output_folder):
    # List of augmentation types
    augmentation_types = ['original', 'blurred', 'scaled', 'rotated', 'brightened', 'noisy']
    
    # Initialize counters for each augmentation type
    counts = {aug: 0 for aug in augmentation_types}
    
    # Ensure output folders exist
    for aug in augmentation_types:
        os.makedirs(os.path.join(output_folder, aug), exist_ok=True)
    
    # Get list of all image files in the input folder
    image_paths = glob(os.path.join(input_folder, '*.png'))
    
    if not image_paths:
        print(f'No images found in {input_folder}')
        return
    
    for image_path in image_paths:
        # Read the image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Process the image
        cuts = cut_image(img)
        if not cuts:
            continue
        
        # Process and save the images
        for i, cut in enumerate(cuts):
            # Save the original cut
            cv2.imwrite(os.path.join(output_folder, 'original', f'image_{i}.png'), cut)
            counts['original'] += 1
            
            # Save augmented images
            augmented_images = {
                'blurred': blur_image(cut),
                'scaled': scale_image(cut),
                'rotated': rotate_image(cut),
                'brightened': change_brightness(cut),
                'noisy': add_noise(cut)
            }
            
            for aug_type, aug_img in augmented_images.items():
                cv2.imwrite(os.path.join(output_folder, aug_type, f'image_{i}.png'), aug_img)
                counts[aug_type] += 1
    
    # Print the counts for each augmentation type
    for aug in augmentation_types:
        print(f'Số lượng ảnh trong thư mục {aug}: {counts[aug]}')

# Example usage
input_folder = 'image/extended sheet/sheet3/sheet3_64x384_gray'
output_folder = 'data/train/x_train'
process_images_in_folder(input_folder, output_folder)
