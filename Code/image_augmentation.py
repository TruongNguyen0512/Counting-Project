import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import random_noise
import os

def augment_and_display(image_path):
    # Check if file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Read the original image
    original = cv2.imread(image_path)
    if original is None:
        raise ValueError(f"Could not load image: {image_path}")
        
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    # Create different augmentations
    # 1. Cutting (Crop)
    h, w = original.shape[:2]
    crop = original[h//4:3*h//4, w//4:3*w//4]
    
    # 2. Blur
    blur = cv2.GaussianBlur(original, (7,7), 0)
    
    # 3. Scaling
    scaled = cv2.resize(original, None, fx=0.5, fy=0.5)
    scaled = cv2.resize(scaled, (original.shape[1], original.shape[0]))
    
    # 4. Rotation
    center = (w//2, h//2)
    rotation_matrix = cv2.getRotationMatrix2D(center, 45, 1.0)
    rotated = cv2.warpAffine(original, rotation_matrix, (w, h))
    
    # 5. Brightness change
    brightness = cv2.convertScaleAbs(original, alpha=1.5, beta=30)
    
    # 6. Noise
    noisy = random_noise(original, mode='gaussian', var=0.01)
    noisy = np.array(255*noisy, dtype='uint8')
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # Display all images
    images = [original, crop, blur, scaled, rotated, brightness, noisy]
    titles = ['Original', 'Cropped', 'Blurred', 'Scaled', 'Rotated', 'Brightness', 'Noisy']
    
    for idx, (img, title) in enumerate(zip(images, titles)):
        ax = fig.add_subplot(2, 4, idx+1)
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Return augmented images for further use if needed
    return {
        'original': original,
        'crop': crop,
        'blur': blur,
        'scaled': scaled,
        'rotated': rotated,
        'brightness': brightness,
        'noisy': noisy
    }

# Example usage
if __name__ == "__main__":
    try:
        # Use the exact path where the file was found
        image_path = "dog.png"
            
        print(f"Attempting to load image from: {image_path}")
        augmented_images = augment_and_display(image_path)
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Please ensure the image exists and the path is correct")
