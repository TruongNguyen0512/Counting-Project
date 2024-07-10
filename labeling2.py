import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_image(image_path):
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

def create_labels(image):
    # Apply GaussianBlur to smooth the image
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Use Sobel operator to find edges
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)
    
    # Take the absolute value of Sobel results
    sobelx = np.abs(sobelx)
    
    # Normalize the result to range [0, 255]
    sobelx = cv2.normalize(sobelx, None, 0, 255, cv2.NORM_MINMAX)
    
    # Threshold to get binary image
    _, binary = cv2.threshold(sobelx, 50, 255, cv2.THRESH_BINARY)
    
    # Create a blank label image
    labels = np.zeros_like(image)
    
    # Sum the binary image along the vertical axis
    vertical_sum = np.sum(binary, axis=0)
    
    # Smooth the vertical sum to get rid of small variations
    vertical_sum_smoothed = cv2.GaussianBlur(vertical_sum.reshape(1, -1), (1, 15), 0).flatten()
    
    # Find peaks in the vertical sum
    peaks = np.where(vertical_sum_smoothed > np.mean(vertical_sum_smoothed))[0]
    
    # Group the peaks into bands
    bands = []
    band_start = peaks[0]
    for i in range(1, len(peaks)):
        if peaks[i] != peaks[i-1] + 1:
            bands.append((band_start, peaks[i-1]))
            band_start = peaks[i]
    bands.append((band_start, peaks[-1]))
    
    # Draw the bands as labels
    for band in bands:
        x_start, x_end = band
        x_center = (x_start + x_end) // 2
        label_width = max(2, (x_end - x_start) // 2)  # Adjust the width to be wider
        x_label_start = x_center - label_width // 2
        x_label_end = x_center + label_width // 2
        cv2.rectangle(labels, (x_label_start, 0), (x_label_end, image.shape[0]), (255), thickness=-1)
    
    return labels

def plot_images(original, labeled):
    fig, axs = plt.subplots(2, 1, figsize=(10, 5))
    axs[0].imshow(original, cmap='gray')
    axs[0].set_title('Original Image')
    axs[0].axis('off')
    
    axs[1].imshow(labeled, cmap='gray')
    axs[1].set_title('Labeled Image')
    axs[1].axis('off')
    
    plt.tight_layout()
    plt.show()

# Load the original image
image_path = 'image/test6.png'
original_image = load_image(image_path)

# Create labels
labeled_image = create_labels(original_image)

# Plot the original and labeled images
plot_images(original_image, labeled_image)

# Save the labeled image
cv2.imwrite('labeled_image.png', labeled_image)
