import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from matplotlib import pyplot as plt


# Функция для применения фильтров
def apply_operator(img, operator_type, lower_thresh, upper_thresh, binarization_thresh):
    img = cv2.equalizeHist(img)
    height, width = img.shape

    if operator_type == 'Кэнни':
        edges = cv2.Canny(img, lower_thresh, upper_thresh)
        return edges

    elif operator_type == 'Собель':
        gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(gx, gy)
        edges = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    elif operator_type == 'Превитт':
        kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
        kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)
        gx = cv2.filter2D(img.astype(np.float32), -1, kernel_x)
        gy = cv2.filter2D(img.astype(np.float32), -1, kernel_y)
        magnitude = cv2.magnitude(gx, gy)
        edges = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    elif operator_type == 'Лапласиан':
        laplacian = cv2.Laplacian(img.astype(np.uint8), cv2.CV_64F, ksize=3)
        edges = cv2.normalize(np.abs(laplacian), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    elif operator_type == 'Кирч':
        kernels = [
            np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]], dtype=np.float32),
            np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]], dtype=np.float32),
            np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]], dtype=np.float32),
            np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]], dtype=np.float32),
            np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]], dtype=np.float32),
            np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]], dtype=np.float32),
            np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]], dtype=np.float32),
            np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]], dtype=np.float32),
        ]
        responses = [cv2.filter2D(img.astype(np.float32), -1, k) for k in kernels]
        max_response = np.maximum.reduce(responses)
        edges = cv2.normalize(max_response, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    else:
        raise ValueError(f"Unsupported operator type: {operator_type}")

    # Бинаризация
    _, edges = cv2.threshold(edges, binarization_thresh, 255, cv2.THRESH_BINARY)
    return edges


# Функция для загрузки изображения
def load_image():
    filename = filedialog.askopenfilename(title="Загрузка изображения",
                                          filetypes=(("Image Files", "*.jpg;*.png;*.bmp"), ("All Files", "*.*")))
    if filename:
        global img
        img = cv2.imread(filename, 0)
        if img is None:
            print("Error: Image not found or invalid format")
            return
        img = cv2.resize(img, (813, 508))
        img_display = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.imshow("Загруженное изображение", img_display)


# Функция для обновления изображения с результатами
def update_image(*args):
    operator = operator_combobox.get()
    lower_thresh = lower_thresh_slider.get()
    upper_thresh = upper_thresh_slider.get()
    binarization_thresh = binarization_thresh_slider.get()

    if img is not None:
        processed_img = apply_operator(img, operator, lower_thresh, upper_thresh, binarization_thresh)

        # Применение маски (например, для полигональной маски)
        polygon_vertices = np.array([[50, 505], [677, 505], [708, 290], [543, 188], [194, 181], [47, 250]], np.int32)
        polygon_mask = np.zeros_like(processed_img)
        cv2.fillPoly(polygon_mask, [polygon_vertices], 255)
        processed_img = cv2.bitwise_and(processed_img, polygon_mask)

        # Показываем результат в том же окне, обновляем изображение
        processed_display = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)
        cv2.imshow(f'{operator} Operator', processed_display)
        cv2.waitKey(1)  # Обновление окна сразу, не закрывается
        cv2.updateWindow(f'{operator} Operator')  # Обновляем окно, чтобы оно не закрывалось


# Создание графического интерфейса
root = tk.Tk()
root.title("Детекторы границ")

# Создание интерфейса
load_button = tk.Button(root, text="Загрузить изображение", command=load_image)
load_button.pack(pady=20)

operator_label = tk.Label(root, text="Выберите метод выделения границ:")
operator_label.pack(pady=10)

operator_combobox = ttk.Combobox(root, values=["Кэнни", "Собель", "Превитт", "Лапласиан", "Кирч"])
operator_combobox.set("Кэнни")  # Значение по умолчанию
operator_combobox.pack(pady=10)

# Ползунки для порогов бинаризации
lower_thresh_label = tk.Label(root, text="Нижний порог бинаризации (Кэнни):")
lower_thresh_label.pack(pady=5)

lower_thresh_slider = tk.Scale(root, from_=0, to_=255, orient="horizontal", length=400, resolution=1,
                               command=update_image)
lower_thresh_slider.set(100)  # Значение по умолчанию для порога нижнего
lower_thresh_slider.pack(pady=5)

upper_thresh_label = tk.Label(root, text="Верхний порог бинаризации (Кэнни):")
upper_thresh_label.pack(pady=5)

upper_thresh_slider = tk.Scale(root, from_=0, to_=255, orient="horizontal", length=400, resolution=1,
                               command=update_image)
upper_thresh_slider.set(200)  # Значение по умолчанию для порога верхнего
upper_thresh_slider.pack(pady=5)

# Ползунок для порога бинаризации
binarization_thresh_label = tk.Label(root, text="Порог бинаризации:")
binarization_thresh_label.pack(pady=5)

binarization_thresh_slider = tk.Scale(root, from_=0, to_=255, orient="horizontal", length=400, resolution=1,
                                      command=update_image)
binarization_thresh_slider.set(128)  # Значение по умолчанию для порога бинаризации
binarization_thresh_slider.pack(pady=5)

process_button = tk.Button(root, text="Применить метод", command=update_image)
process_button.pack(pady=20)

# Главный цикл
root.mainloop()
