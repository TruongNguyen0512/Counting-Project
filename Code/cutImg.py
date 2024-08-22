from PIL import Image
import os

# Load the image using the absolute path
image_path = r'D:\Uni\Đồ án tốt nghiệp\Counting Argorithm\image\extended sheet\sheet3\1.png'  # Use raw string to handle backslashes
image = Image.open(image_path)

# Get the dimensions of the image
img_width, img_height = image.size

# Define the dimensions of the segments
segment_width = 64
segment_height = 384

# Create a directory to save the segments
output_dir = r'D:\Uni\Đồ án tốt nghiệp\Counting Argorithm\image\extended sheet\sheet3\sheet3_64x384'
os.makedirs(output_dir, exist_ok=True)

# Calculate the number of segments
num_segments = img_width // segment_width

# Extract and save each segment
for i in range(num_segments):
    left = i * segment_width
    right = (i + 1) * segment_width
    top = 0
    bottom = segment_height

    segment = image.crop((left, top, right, bottom))
    segment_path = os.path.join(output_dir, f'segment_{i}.png')
    segment.save(segment_path)

# Confirm completion
print(f'Segments saved to: {output_dir}')
