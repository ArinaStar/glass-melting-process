import cv2
import numpy as np

# Загружаем изображение
img = cv2.imread("For_Click.jpg")
img = cv2.resize(img, (1336, 779))
print(img.shape)

# Исходные точки (с изображения)
src_pts = np.float32([
    [215, 201],
    [630, 203],
    [803, 307],
    [735, 574],
    [105, 574],
    [53, 304]
])

# Целевые точки (как они должны выглядеть сверху)
dst_pts = np.float32([
    [100, 0],
    [400, 0],
    [500, 200],
    [400, 500],
    [100, 500],
    [0, 200]
])

# Находим гомографию
H, _ = cv2.findHomography(src_pts, dst_pts)

# Размер итогового изображения (ширина, высота)
output_size = (800, 700)
print(img.shape)

# Применяем преобразование
warped = cv2.warpPerspective(img, H, output_size)

# Показываем результат
cv2.imshow("Top-down View", warped)
cv2.imshow("IMG", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Сохраняем при необходимости
cv2.imwrite("top_view_result.jpg", warped)