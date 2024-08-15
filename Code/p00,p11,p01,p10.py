import cv2
import numpy as np

def resize_to_match(original, segmented):
    # Kiểm tra nếu kích thước không khớp, thay đổi kích thước của segmented để khớp với original
    if original.shape != segmented.shape:
        segmented = cv2.resize(segmented, (original.shape[1], original.shape[0]))
    return segmented

def count_pixels(original, segmented):
    # Thay đổi kích thước của segmented để khớp với original nếu cần thiết
    segmented = resize_to_match(original, segmented)
    
    # Chuyển đổi hình ảnh thành nhị phân
    _, original_binary = cv2.threshold(original, 127, 255, cv2.THRESH_BINARY)
    _, segmented_binary = cv2.threshold(segmented, 127, 255, cv2.THRESH_BINARY)
    
    # Xác định các giá trị pixel
    p00 = np.sum((original_binary == 0) & (segmented_binary == 0))
    p01 = np.sum((original_binary == 0) & (segmented_binary == 255))
    p11 = np.sum((original_binary == 255) & (segmented_binary == 255))
    p10 = np.sum((original_binary == 255) & (segmented_binary == 0))
    
    return p00, p01, p11, p10

# Đọc hình ảnh
original_image = cv2.imread('image/test9.png', cv2.IMREAD_GRAYSCALE)
segmented_image_1 = cv2.imread('image/test9.png', cv2.IMREAD_GRAYSCALE)
segmented_image_2 = cv2.imread('image/test10.png', cv2.IMREAD_GRAYSCALE)

# Đếm pixel cho hình ảnh phân đoạn 1
p00_1, p01_1, p11_1, p10_1 = count_pixels(original_image, segmented_image_1)
print(f"Hình ảnh phân đoạn 1: p00={p00_1}, p01={p01_1}, p11={p11_1}, p10={p10_1}")

# Đếm pixel cho hình ảnh phân đoạn 2
p00_2, p01_2, p11_2, p10_2 = count_pixels(original_image, segmented_image_2)
print(f"Hình ảnh phân đoạn 2: p00={p00_2}, p01={p01_2}, p11={p11_2}, p10={p10_2}")
