import os

# specify the folder path
folder_path = r"D:\Uni\Đồ án tốt nghiệp\Counting Argorithm\labeled-Vu"

# loop through all files in the folder
for filename in os.listdir(folder_path):
    # check if the file is an image (assuming it has a .jpg or .png extension)
    if filename.endswith(('.jpg', '.png')):
        # check if the filename starts with "segment_"
        if filename.startswith('segment_'):
            # remove the "segment_" prefix
            new_filename = filename.replace('segment_', '', 1)
            # construct the full new filename
            new_full_filename = os.path.join(folder_path, new_filename)
            # rename the file
            os.rename(os.path.join(folder_path, filename), new_full_filename)
