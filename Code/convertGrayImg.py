import os
from PIL import Image

# Define the directories with absolute paths
input_dir = r'D:\Uni\Đồ án tốt nghiệp\Counting Argorithm\image\extended sheet\sheet3\sheet3_64x384'
output_dir = r'D:\Uni\Đồ án tốt nghiệp\Counting Argorithm\image\extended sheet\sheet3\sheet3_64x384_gray'

# Create the output directory if it does not exist
os.makedirs(output_dir, exist_ok=True)

# Process each image in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(".png"):
        # Load the image
        img_path = os.path.join(input_dir, filename)
        image = Image.open(img_path)

        # Convert to grayscale
        gray_image = image.convert('L')

        # Save the grayscale image
        gray_image_path = os.path.join(output_dir, filename)
        gray_image.save(gray_image_path)

print(f'Grayscale images saved to: {output_dir}')
