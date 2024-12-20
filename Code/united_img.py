import os
import shutil

# Đường dẫn đến 7 folder ảnh xám
folders = [   
    'image/labeled sheet/sheet1',
    'image/labeled sheet/sheet2',
    'image/labeled sheet/sheet3',
    'image/labeled sheet/sheet4',
    'image/labeled sheet/sheet5',
    'image/labeled sheet/sheet6',
    'image/labeled sheet/sheet7',
]

# Đường dẫn đến folder mới để chứa tất cả ảnh
destination_folder = 'image/united imge labeled'

# Tạo folder đích nếu chưa tồn tại
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# Khởi tạo biến đếm để đánh số thứ tự cho ảnh
count = 0

# Lặp qua tất cả các folder và sao chép ảnh vào folder mới
for folder in folders:
    for filename in os.listdir(folder):
        if filename.endswith('.png'):  # Lọc chỉ ảnh định dạng .png (hoặc định dạng khác nếu cần)
            source_path = os.path.join(folder, filename)
            new_filename = f"label_{count}.png"  # Đặt tên mới cho ảnh
            destination_path = os.path.join(destination_folder, new_filename)
            shutil.copy(source_path, destination_path)  # Sao chép ảnh vào folder mới
            count += 1  # Tăng biến đếm

print(f"Tổng số {count} ảnh đã được gộp vào folder {destination_folder}")
