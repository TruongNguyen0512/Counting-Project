import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load the newly uploaded image
new_image_path = 'image/test5.png'
new_image = Image.open(new_image_path)
new_image = np.array(new_image)

# Convert to grayscale
gray_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)

# Apply a vertical Sobel filter to detect vertical edges (ridge lines)
sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5)
abs_sobelx = np.absolute(sobelx)
sobelx_8u = np.uint8(abs_sobelx)

# Threshold the Sobel image to get binary image of the edges
_, binary_image = cv2.threshold(sobelx_8u, 50, 255, cv2.THRESH_BINARY)

# Find contours of the binary image
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create a mask for labeling
mask = np.zeros_like(gray_image)

# Draw filled rectangles around the ridge lines (contours)
for contour in contours:
    if cv2.contourArea(contour) > 100:  # Filter out small contours
        x, y, w, h = cv2.boundingRect(contour)
        # Draw a filled rectangle along the ridge line
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

# Create binary masks for bright and dark stripes
bright_stripes = np.zeros_like(gray_image)
dark_stripes = np.zeros_like(gray_image)

# Define the criteria for bright and dark stripes
threshold_value = 100
_, bright_stripes = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)
dark_stripes = cv2.bitwise_not(bright_stripes)

# Combine the masks with the original image
combined_image = np.vstack((gray_image, bright_stripes, dark_stripes))

# Display the combined image
plt.figure(figsize=(12, 6))
plt.imshow(combined_image, cmap='gray')
plt.axis('off')
plt.show()
