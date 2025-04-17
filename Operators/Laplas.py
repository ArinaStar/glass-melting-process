import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
from PIL import Image
from PIL import ImageDraw
from PIL import ImageColor

IMAGE_NAME = '../data/Fire_pyatna.jpg'

temp_img = cv2.imread(IMAGE_NAME, 0)
temp_img = cv2.resize(temp_img, (813, 508))
img = cv2.equalizeHist(temp_img)


# Применение оператора Лапласа (готовая функция)
laplacian = cv2.Laplacian(img, cv2.CV_64F, ksize=3)
img2 = np.abs(laplacian)  # Модуль значений (градиенты могут быть отрицательными)

img2 = cv2.normalize(img2, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Применяем бинаризацию
img2[img2 < 5] = 0
img2[img2 >= 5] = 255

# --- НАЛОЖЕНИЕ ПОЛИГОНА ---
height, width = img.shape
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

plt.show()