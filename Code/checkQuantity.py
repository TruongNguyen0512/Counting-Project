import os
from PIL import Image

def is_image_file(file_path):
    try:
        with Image.open(file_path) as img:
            return True
    except:
        return False

def count_images_in_folders(parent_folder):
    folder_image_count = {}

    # Duyệt qua tất cả các thư mục con
    for root, dirs, files in os.walk(parent_folder):
        image_count = 0

        # Kiểm tra từng file trong thư mục con
        for file in files:
            file_path = os.path.join(root, file)

            if is_image_file(file_path):
                image_count += 1

        folder_image_count[root] = image_count

    return folder_image_count

# Đường dẫn đến thư mục cha
parent_folder_path = "data/train/y_train"
image_counts = count_images_in_folders(parent_folder_path)

# In kết quả
for folder, count in image_counts.items():
    print(f"Thư mục: {folder} có {count} ảnh")
