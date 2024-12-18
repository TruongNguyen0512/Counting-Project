import os
import shutil

# Define the source folders and the destination folder
source_folders = [
    'labeled-Truong',
    'labeled-Thang',    
    'labeled-Vu',
]
destination_folder = 'Asem-labeled-Sample'

# Create the destination folder if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

# Function to copy images from a source folder to the destination folder
def copy_images(source_folder, destination_folder):
    for filename in os.listdir(source_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            source_path = os.path.join(source_folder, filename)
            destination_path = os.path.join(destination_folder, filename)
            shutil.copy2(source_path, destination_path)
            print(f"Copied: {filename}")

# Copy images from each source folder to the destination folder
for folder in source_folders:
    print(f"Processing folder: {folder}")
    copy_images(folder, destination_folder)

print("Image assembly completed!")
