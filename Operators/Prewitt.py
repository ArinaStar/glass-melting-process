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
img = temp_img.astype(np.int32)  # Чтобы избежать переполнения при умножении

img2 = cv2.equalizeHist(img.astype(np.uint8)).astype(np.int32)  # OpenCV требует uint8

width = len(img[0])
height = len(img)

gx = [
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
]

gy = [
    [1, 1, 1],
    [0, 0, 0],
    [-1, -1, -1]
]

for i in range(1, width - 1):
    for j in range(1, height - 1):
        gradientX = (
                gx[0][0] * img[j - 1][i - 1] +
                gx[0][2] * img[j - 1][i + 1] +
                gx[1][0] * img[j][i - 1] +
                gx[1][2] * img[j][i + 1] +
                gx[2][0] * img[j + 1][i - 1] +
                gx[2][2] * img[j + 1][i + 1]
        )

        gradientY = (
                gy[0][0] * img[j - 1][i - 1] +
                gy[0][1] * img[j - 1][i] +
                gy[0][2] * img[j - 1][i + 1] +
                gy[2][0] * img[j + 1][i - 1] +
                gy[2][1] * img[j + 1][i] +
                gy[2][2] * img[j + 1][i + 1]
        )

        gradientMagn = math.sqrt(gradientX * gradientX + gradientY * gradientY)
        img2[j][i] = gradientMagn

img2 = cv2.normalize(img2, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Применяем бинаризацию
img2[img2 < 7] = 0
img2[img2 >= 7] = 255

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

plt.show()