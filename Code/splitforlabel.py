import os
import shutil
import math

def split_folder_into_parts(input_folder, num_parts=3):
    # Create output folders
    base_path = os.path.dirname(input_folder)
    folder_name = os.path.basename(input_folder)
    
    # Get all image files
    image_files = []
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    for file in os.listdir(input_folder):
        if file.lower().endswith(valid_extensions):
            image_files.append(file)
    
    # Calculate images per folder
    total_images = len(image_files)
    images_per_folder = math.ceil(total_images / num_parts)
    
    # Create and fill output folders
    for i in range(num_parts):
        # Create folder
        output_folder = os.path.join(base_path, f"{folder_name}_part{i+1}")
        os.makedirs(output_folder, exist_ok=True)
        
        # Calculate start and end indices for this part
        start_idx = i * images_per_folder
        end_idx = min((i + 1) * images_per_folder, total_images)
        
        # Copy images to the output folder
        for file in image_files[start_idx:end_idx]:
            src = os.path.join(input_folder, file)
            dst = os.path.join(output_folder, file)
            shutil.copy2(src, dst)
        
        print(f"Created {output_folder} with {end_idx - start_idx} images")

# Use the function
input_folder = r'D:\Uni\Đồ án tốt nghiệp\Counting Argorithm\output_64x384'
split_folder_into_parts(input_folder)