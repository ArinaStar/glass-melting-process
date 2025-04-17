import cv2
import numpy as np

IMAGE_NAME = '../data/Fire_pyatna.jpg'  # Путь к изображению

# Загружаем изображение и изменяем размер
temp = cv2.imread(IMAGE_NAME, 0)
temp = cv2.resize(temp, (813, 508))

img = cv2.imread(IMAGE_NAME, 0)
img = cv2.resize(img, (813, 508))

# Улучшение контраста
img = cv2.equalizeHist(img)

# Применяем детектор Canny
low_threshold = 60  # Нижний порог
high_threshold = 110  # Верхний порог
edges = cv2.Canny(img, low_threshold, high_threshold)

# --- НАЛОЖЕНИЕ ПОЛИГОНА ---
height, width = edges.shape
polygon_vertices = np.array([
    [50, 505],
    [677, 505],
    [708, 290],
    [543, 188],
    [194, 181],
    [47, 250]
], np.int32)

polygon_mask = np.zeros((height, width), dtype=np.uint8)
cv2.fillPoly(polygon_mask, [polygon_vertices], 255)

edges_masked = cv2.bitwise_and(edges, polygon_mask)

# Поиск контуров
cnts, _ = cv2.findContours(edges_masked, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Рисуем контуры на оригинальном изображении
image_with_contours = cv2.cvtColor(temp, cv2.COLOR_GRAY2BGR)
cv2.drawContours(image_with_contours, cnts, -1, (0, 0, 255), 1)  # Красные контуры

# Отображение изображений
cv2.imshow('Original Image', temp)
cv2.imshow('Canny Edges (Masked)', edges_masked)
cv2.imshow('Contours on Image', image_with_contours)

cv2.waitKey(0)
cv2.destroyAllWindows()