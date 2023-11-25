import os
import sys

import cv2
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QTextEdit, QFileDialog
from PyQt5.QtCore import Qt
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

result_class_dict = {"bad": "Не распознано", "beton": "Бетон", "brick": "Кирпичи", "dirt": "Грунт", "tree": "Деревья"}


def recognition(file_path):
    frame_rate = 1
    model = tf.keras.models.load_model('model1.h5')
    # preds = loaded_model.predict(x)

    cap = cv2.VideoCapture(file_path)

    # Получаем частоту кадров (fps) видео
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Устанавливаем интервал между кадрами для сохранения
    interval = int(fps / frame_rate)

    # Устанавливаем позицию видеофайла в начальное время
    cap.set(cv2.CAP_PROP_POS_MSEC, 110000)

    # Считываем и сохраняем кадры в заданном временном диапазоне
    frame_count = 0
    result_dict = {"bad": 0, "beton": 0, "brick": 0, "dirt": 0, "tree": 0}
    class_names = {0: "bad", 1: "beton", 2: "brick", 3: "dirt", 4: "tree"}
    result_preds = np.array([[0 for i in range(5)]])
    while True:
        ret, frame = cap.read()

        if not ret or cap.get(cv2.CAP_PROP_POS_MSEC) > 140000:
            break

        frame_count += 1
        if frame_count % interval == 0:
            frame = cv2.resize(frame, (224, 224))

            x = image.img_to_array(frame)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            preds = model.predict(x)

            pred_num = np.argmax(preds)
            result_dict[class_names[pred_num]] += 1
            result_preds = result_preds + preds
        # print()
    # Закрываем видеофайл
    cap.release()
    # print(result_dict)
    # print(result_preds)
    result = class_names[np.argmax(result_preds[:, 1:]) + 1]
    return result_class_dict[result]


class MyWindow(QWidget):



    def __init__(self):
        super().__init__()

        # Создаем элементы интерфейса
        self.initUI()

    def initUI(self):
        # Кнопка
        button = QPushButton('Выбрать видео', self)
        button.clicked.connect(self.on_button_click)

        # Поле для вывода текста (для отображения выбранного файла)
        self.file_label = QLabel(self)
        self.file_label.setAlignment(Qt.AlignCenter)

        # Создаем вертикальный лейаут и добавляем в него элементы
        vbox = QVBoxLayout()
        vbox.addWidget(button)
        vbox.addWidget(self.file_label)

        # Устанавливаем лейаут в основное окно
        self.setLayout(vbox)

        # Устанавливаем размеры окна
        self.setGeometry(100, 100, 200, 200)
        self.setWindowTitle('Пример PyQt GUI')
        self.show()

    def on_button_click(self):
        self.file_label.setText("")
        # Открываем диалог выбора файла видео
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("Videos (*.mp4 *.avi *.mkv)")
        file_path, _ = file_dialog.getOpenFileName(self, 'Выбрать видео', '', 'Videos (*.mp4 *.avi *.mkv)')

        # Если пользователь выбрал файл, отображаем его имя
        if file_path:
            self.file_label.setText(recognition(file_path))
            # print(file_path)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyWindow()
    sys.exit(app.exec_())
