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

# Function to process all images in a folder
def process_images_in_folder(input_folder, output_folder):
    # Get list of all image files in the input folder
    image_paths = glob(os.path.join(input_folder, '*.png'))
    
    if not image_paths:
        print(f'No images found in {input_folder}')
    
    for image_path in image_paths:
        # Load the image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f'Error loading image {image_path}')
            continue
        
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
        
        # Create output directory for the current image
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        image_output_dir = os.path.join(output_folder, image_name)
        os.makedirs(image_output_dir, exist_ok=True)
        
        # Save the augmented images
        for i, aug_img in enumerate(augmented_images):
            output_file_path = os.path.join(image_output_dir, f'augmented_{i}.png')
            cv2.imwrite(output_file_path, aug_img)
        
        print(f'Processed and generated {len(augmented_images)} augmented images for {image_name}.')

# Specify the input and output folders
input_folders = [
    'image/extended sheet/sheet1/sheet1_64x384_gray',
    'image/extended sheet/sheet2/sheet2_64x384_gray',
    'image/extended sheet/sheet3/sheet3_64x384_gray',
    'image/extended sheet/sheet4/sheet4_64x384_gray',
    'image/extended sheet/sheet5/sheet5_64x384_grday',
    'image/extended sheet/sheet6/sheet6_64x384_gray',
    'image/extended sheet/sheet7/sheet7_64x384_gray' 
]

output_folder = 'data/train/x_train'

# Process all images in each folder
for input_folder in input_folders:
    if not os.path.exists(input_folder):
        print(f'Input folder does not exist: {input_folder}')
    else:
        process_images_in_folder(input_folder, output_folder)
