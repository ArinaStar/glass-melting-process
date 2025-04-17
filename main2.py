import sys
import cv2
import scipy
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QWidget, QHBoxLayout
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtGui import QFont

import skimage
from skimage.measure import label
from skimage import img_as_float64
from skimage import restoration

default_polygon = np.array([
    [215, 113],
    [625, 118],
    [733, 270],
    [577, 529],
    [144, 530],
    [54, 277]
], np.int32)


def apply_perspective_transform(frame):
    # src_pts = np.float32([
    #     [263, 272],
    #     [771, 275],
    #     [982, 415],
    #     [900, 776],
    #     [129, 776],
    #     [65, 411]
    # ])
    # dst_pts = np.float32([
    #     [122, 0],
    #     [612, 0],
    #     [612, 270],
    #     [490, 676],
    #     [122, 676],
    #     [0, 270]
    # ])
    dst_pts = np.float32([
        [100, 0],
        [400, 0],
        [500, 200],
        [400, 500],
        [100, 500],
        [0, 200]
    ])
    src_pts = np.float32([
        [215, 201],
        [630, 203],
        [803, 307],
        [735, 574],
        [105, 574],
        [53, 304]
    ])
    H, _ = cv2.findHomography(src_pts, dst_pts)
    warped = cv2.warpPerspective(frame, H, (800, 700))
    return warped


def process_frame(frame, polygon_vertices, info_label=None):
    # Предобработка изображения
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_color = frame.copy()
    img = cv2.resize(img, (928, 576))
    img_color = cv2.resize(img_color, (928, 576))

    img_hist_eq = cv2.equalizeHist(img)
    img_float = img_as_float64(img_hist_eq)
    kernel = np.ones((15, 15), np.float64)
    image_filtered = scipy.signal.convolve2d(img_float, kernel, 'same')
    img_wiener = skimage.restoration.wiener(image_filtered, kernel, 5.1e4)
    img_wiener = skimage.img_as_ubyte(img_wiener)

    image_filtered = skimage.img_as_ubyte(img_wiener)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    image_filtered = clahe.apply(image_filtered)

    polygon_mask = np.zeros_like(img, dtype=np.uint8)
    cv2.fillPoly(polygon_mask, [polygon_vertices], 255)

    thresh = cv2.adaptiveThreshold(image_filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 191, 0)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    thresh = cv2.bitwise_and(thresh, polygon_mask)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_color, contours, -1, (0, 255, 0), 1)
    cv2.polylines(img_color, [polygon_vertices], True, (255, 0, 0), 2)

    mid_x = (polygon_vertices[:, 0].min() + polygon_vertices[:, 0].max()) // 2
    min_y = polygon_vertices[:, 1].min()
    max_y = polygon_vertices[:, 1].max()
    cv2.line(img_color, (mid_x, min_y), (mid_x, max_y), (0, 0, 255), 2)

    # Полигон на левую и правую части
    left_polygon = []
    right_polygon = []
    for pt in polygon_vertices:
        if pt[0] < mid_x:
            left_polygon.append(pt)
        else:
            right_polygon.append(pt)
    left_polygon = np.array(left_polygon, np.int32)
    right_polygon = np.array(right_polygon, np.int32)

    left_mask = np.zeros_like(img, dtype=np.uint8)
    right_mask = np.zeros_like(img, dtype=np.uint8)
    cv2.fillPoly(left_mask, [left_polygon], 255)
    cv2.fillPoly(right_mask, [right_polygon], 255)

    left_thresh = cv2.bitwise_and(thresh, left_mask)
    right_thresh = cv2.bitwise_and(thresh, right_mask)
    left_contours, _ = cv2.findContours(left_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    right_contours, _ = cv2.findContours(right_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    left_area = sum(cv2.contourArea(cnt) for cnt in left_contours)
    right_area = sum(cv2.contourArea(cnt) for cnt in right_contours)

    # Суммарная площадь контуров в левой и правой частях
    total_contour_area = left_area + right_area
    if total_contour_area > 0:
        left_percent = (left_area / total_contour_area) * 100
        right_percent = (right_area / total_contour_area) * 100
    else:
        left_percent = right_percent = 0

    if info_label:
        info_label.setText(
            f"Левая часть: {left_percent:.2f}%, Правая часть: {right_percent:.2f}%"
        )

    return img_color

class PolygonLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.polygon_vertices = default_polygon.copy()
        self.dragging_point = None
        self.is_editing = False
        self.image = None

    def set_image(self, image):
        self.image = image.copy()
        self.update_display()

    def set_polygon(self, polygon):
        self.polygon_vertices = polygon.copy()

    def get_polygon(self):
        return self.polygon_vertices.copy()

    def update_display(self):
        if self.image is None:
            return
        base = self.image.copy()

        if self.is_editing:
            cv2.polylines(base, [self.polygon_vertices], True, (255, 0, 0), 2)
            for pt in self.polygon_vertices:
                cv2.circle(base, tuple(pt), 5, (0, 255, 0), -1)

        img = cv2.cvtColor(base, cv2.COLOR_BGR2RGB)
        q_img = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(q_img).scaled(self.size(), Qt.KeepAspectRatio))

    def mousePressEvent(self, event):
        if not self.is_editing:
            return
        pos = self._map_mouse_to_image(event.pos())
        for i, pt in enumerate(self.polygon_vertices):
            if np.linalg.norm(np.array(pos) - pt) < 10:
                self.dragging_point = i
                break

    def mouseMoveEvent(self, event):
        if self.dragging_point is not None and self.is_editing:
            pos = self._map_mouse_to_image(event.pos())
            self.polygon_vertices[self.dragging_point] = pos
            self.update_display()

    def mouseReleaseEvent(self, event):
        self.dragging_point = None

    def _map_mouse_to_image(self, pos):
        label_width = self.width()
        label_height = self.height()
        img_height, img_width = self.image.shape[:2]

        ratio = min(label_width / img_width, label_height / img_height)
        offset_x = (label_width - img_width * ratio) / 2
        offset_y = (label_height - img_height * ratio) / 2

        x = int((pos.x() - offset_x) / ratio)
        y = int((pos.y() - offset_y) / ratio)
        return [x, y]


class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Анализ изображения")

        self.btn_load = QPushButton("Загрузить изображение")
        self.btn_load.clicked.connect(self.load_image)

        self.btn_edit = QPushButton("Изменить полигон")
        self.btn_edit.clicked.connect(self.toggle_polygon_editing)
        self.label_info = QLabel("")
        self.label_info.setAlignment(Qt.AlignCenter)
        self.label_info.setFixedHeight(100)
        font = QFont("Arial", 18, QFont.Bold)
        self.label_info.setFont(font)

        self.label_original = QLabel("Оригинальное изображение")
        self.label_processed = PolygonLabel()

        layout = QVBoxLayout()
        image_layout = QHBoxLayout()
        image_layout.addWidget(self.label_original)
        image_layout.addWidget(self.label_processed)

        layout.addWidget(self.btn_load)
        layout.addWidget(self.btn_edit)
        layout.addWidget(self.label_info)
        layout.addLayout(image_layout)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.edit_mode = False
        self.last_frame = None
        self.current_polygon = default_polygon.copy()

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Выберите изображение")
        if not file_path:
            return

        img = cv2.imread(file_path)
        if img is None:
            return
        img2 = cv2.resize(img, (928, 576))
        img = apply_perspective_transform(cv2.resize(img, (1136, 779)))
        self.last_frame = cv2.resize(img, (928, 576))

        self.original_img = img2
        self.display_image(self.original_img, self.label_original)

        processed = process_frame(self.last_frame.copy(), self.current_polygon, self.label_info)
        self.label_processed.set_image(processed)
        self.label_processed.set_polygon(self.current_polygon)
        self.label_processed.is_editing = False
        self.label_processed.update_display()

    def toggle_polygon_editing(self):
        self.edit_mode = not self.edit_mode

        if self.edit_mode:
            self.btn_edit.setText("Завершить редактирование")
            self.label_processed.set_image(self.last_frame.copy())
            self.label_processed.set_polygon(self.current_polygon)
            self.label_processed.is_editing = True
            self.label_processed.update_display()
        else:
            self.btn_edit.setText("Изменить полигон")
            self.current_polygon = self.label_processed.get_polygon()
            processed = process_frame(self.last_frame.copy(), self.current_polygon, self.label_info)
            self.label_processed.set_image(processed)
            self.label_processed.is_editing = False
            self.label_processed.update_display()

    def display_image(self, img, label):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        q_img = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        label.setPixmap(pixmap.scaled(label.size(), Qt.KeepAspectRatio))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainApp()
    window.resize(1900, 1000)
    window.show()
    sys.exit(app.exec_())
