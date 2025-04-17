import cv2
import numpy as np

IMAGE_NAME = '../data/Fire_pyatna.jpg'  # Укажи путь к изображению

temp_img = cv2.imread(IMAGE_NAME, 0)
temp_img = cv2.resize(temp_img, (813, 508))
img = temp_img.astype(np.int32)  # Чтобы избежать переполнения при умножении

img2 = cv2.equalizeHist(img.astype(np.uint8)).astype(np.int32)  # OpenCV требует uint8

height, width = img.shape

# Определим ядра Кирша
kirsch_kernels = [
    np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]], dtype=np.int32),
    np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]], dtype=np.int32),
    np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]], dtype=np.int32),
    np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]], dtype=np.int32),
    np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]], dtype=np.int32),
    np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]], dtype=np.int32),
    np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]], dtype=np.int32),
    np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]], dtype=np.int32),
]

# Применим фильтры и возьмём максимум из откликов
responses = [cv2.filter2D(img.astype(np.float32), -1, k) for k in kirsch_kernels]
img2 = np.max(np.stack(responses, axis=0), axis=0)

# Нормализация
img2 = cv2.normalize(img2, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Применяем бинаризацию
_, img2 = cv2.threshold(img2, 5, 70, cv2.THRESH_BINARY)

# --- НАЛОЖЕНИЕ ПОЛИГОНА ---
polygon_vertices = np.array([
    [50, 505], [677, 505], [708, 290], [543, 188], [194, 181], [47, 250]
], np.int32)

polygon_mask = np.zeros((height, width), dtype=np.uint8)
cv2.fillPoly(polygon_mask, [polygon_vertices], 255)

img2 = cv2.bitwise_and(img2, polygon_mask)

cnts, _ = cv2.findContours(img2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

image_with_contours = cv2.cvtColor(temp_img, cv2.COLOR_GRAY2BGR)
cv2.drawContours(image_with_contours, cnts, -1, (0, 0, 255), 1)

cv2.imshow('Original Image', img.astype(np.uint8))
cv2.imshow('Enhanced Kirsch Operator (Masked)', img2)
cv2.imshow('Contours on Image', image_with_contours)

cv2.waitKey(0)
cv2.destroyAllWindows()
