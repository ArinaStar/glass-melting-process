import cv2
import numpy as np
import math
from matplotlib import pyplot as plt

IMAGE_NAME = '../data/Fire_pyatna.jpg'

# Загружаем изображение
temp = cv2.imread(IMAGE_NAME, 0)
temp = cv2.resize(temp, (813, 508))

img = cv2.imread(IMAGE_NAME, 0)
img = cv2.equalizeHist(img)  # Улучшаем контраст
img = cv2.resize(img, (813, 508)).astype(np.int32)

height, width = img.shape

# Создаём пустое изображение для градиентов
img2 = np.zeros_like(img, dtype=np.float32)

# Операторы Собеля
gx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
gy = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]

# Границы значений градиента
gradientMax, gradientMin = -1000, 1000

# Вычисление градиентов
for i in range(1, width - 1):
    for j in range(1, height - 1):
        gradientX = (gx[0][0] * img[j - 1][i - 1] + gx[0][2] * img[j - 1][i + 1] +
                     gx[1][0] * img[j][i - 1] + gx[1][2] * img[j][i + 1] +
                     gx[2][0] * img[j + 1][i - 1] + gx[2][2] * img[j + 1][i + 1])

        gradientY = (gy[0][0] * img[j - 1][i - 1] + gy[0][1] * img[j - 1][i] +
                     gy[0][2] * img[j - 1][i + 1] + gy[2][0] * img[j + 1][i - 1] +
                     gy[2][1] * img[j + 1][i] + gy[2][2] * img[j + 1][i + 1])

        gradientMagn = math.sqrt(gradientX ** 2 + gradientY ** 2)
        img2[j][i] = gradientMagn

        gradientMax = max(gradientMax, gradientMagn)
        gradientMin = min(gradientMin, gradientMagn)

# Нормализация градиентов
img2 = cv2.normalize(img2, None, 0, 255, cv2.NORM_MINMAX)
img2 = img2.astype(np.uint8)

# Применяем бинаризацию
img2[img2 < 15] = 0
img2[img2 >= 15] = 255

### === Добавляем полигон === ###
polygon_vertices = np.array([
    [50, 505],
    [677, 505],
    [708, 290],
    [543, 188],
    [194, 181],
    [47, 250]
], np.int32)

# Создаём маску полигона
polygon_mask = np.zeros((height, width), dtype=np.uint8)
cv2.fillPoly(polygon_mask, [polygon_vertices], 255)

# Оставляем контуры только внутри полигона
img2  = cv2.bitwise_and(img2, polygon_mask)

cnts, _ = cv2.findContours(img2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

image_with_contours = cv2.cvtColor(temp, cv2.COLOR_GRAY2BGR)
cv2.drawContours(image_with_contours, cnts, -1, (0, 0, 255), 1)  # Красные контуры

# Вывод изображений в OpenCV
cv2.imshow('Original Image', img.astype(np.uint8))
cv2.imshow('Edges Inside Polygon', img2)
cv2.imshow('Image with contours', image_with_contours)

cv2.waitKey(0)
cv2.destroyAllWindows()