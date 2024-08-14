import cv2
import numpy as np
import matplotlib.pyplot as plt

def morphological_skeletonization(image_path, output_path):
    # Tải hình ảnh nhị phân
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        raise ValueError(f"Không thể mở hoặc đọc file: {image_path}")
    
    # Chuyển đổi ảnh thành nhị phân
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    
    # Tạo kernel cho xói mòn và phồng rộng
    kernel = np.ones((3, 3), np.uint8)
    
    # Tạo ảnh xương rỗng
    skeleton = np.zeros_like(binary_image)
    
    # Thực hiện xói mòn và phồng rộng cho đến khi không còn thay đổi
    while True:
        # Xói mòn
        eroded = cv2.erode(binary_image, kernel)
        
        # Phồng rộng
        dilated = cv2.dilate(eroded, kernel)
        
        # Tìm phần xương mới
        skeleton_part = cv2.subtract(binary_image, dilated)
        
        # Cập nhật xương
        skeleton = cv2.bitwise_or(skeleton, skeleton_part)
        
        # Cập nhật ảnh nhị phân
        binary_image = eroded
        
        # Dừng khi không còn thay đổi
        if cv2.countNonZero(binary_image) == 0:
            break
    
    # Lưu ảnh xương
    cv2.imwrite(output_path, skeleton)
    
    # Hiển thị kết quả
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(image, cmap='gray')
    
    plt.subplot(1, 2, 2)
    plt.title('Skeletonized Image')
    plt.imshow(skeleton, cmap='gray')
    
    plt.show()

# Ví dụ sử dụng
image_path = 'image/test7.png'
output_path = 'skeletonized_image.png'
morphological_skeletonization(image_path, output_path)
