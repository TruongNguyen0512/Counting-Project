import sys
import os
from PIL import Image

# Reconfigure stdout to use UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# Define the input directories
input_dirs = [
    r'D:\Uni\Đồ án tốt nghiệp\Counting Argorithm\images\newType1',
    r'D:\Uni\Đồ án tốt nghiệp\Counting Argorithm\images\newType2',
    r'D:\Uni\Đồ án tốt nghiệp\Counting Argorithm\images\newType3',
    r'D:\Uni\Đồ án tốt nghiệp\Counting Argorithm\images\newType4'
]

# Define the output directory
output_dir = r'D:\Uni\Đồ án tốt nghiệp\Counting Argorithm\output_64x384'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Define the dimensions of the segments
segment_width = 64
segment_height = 384

# Initialize a counter for segment naming
segment_counter = 0

try:
    # Process each directory
    for input_dir in input_dirs:
        print(f'Processing directory: {input_dir}')
        
        # Process each image in the directory
        for image_name in os.listdir(input_dir):
            image_path = os.path.join(input_dir, image_name)
            
            # Check if the path is a file and has a valid image extension
            if os.path.isfile(image_path) and image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                try:
                    # Load the image
                    image = Image.open(image_path)

                    # Get the dimensions of the image
                    img_width, img_height = image.size

                    # Calculate the number of segments
                    num_segments = img_width // segment_width

                    # Extract and save each segment
                    for i in range(num_segments):
                        left = i * segment_width
                        right = (i + 1) * segment_width
                        top = 0
                        bottom = segment_height

                        segment = image.crop((left, top, right, bottom))
                        segment_path = os.path.join(output_dir, f'segment_{segment_counter:04d}.png')
                        segment.save(segment_path)
                        segment_counter += 1
                        
                    print(f'Processed: {image_name} - Created {num_segments} segments')
                    
                except Exception as e:
                    print(f"Error processing image {image_name}: {e}")
                    continue

    print(f'\nProcessing complete!')
    print(f'Total segments created: {segment_counter}')
    print(f'All segments saved to: {output_dir}')
    
except Exception as e:
    print(f"An error occurred: {e}")