import sys
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import math
import tensorflow as tf
from tensorflow import lite
from PyQt5.QtGui import QImage, QPixmap, QIcon, QFont
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget, QLineEdit, QPushButton, QHBoxLayout, QDialog, QMenuBar, QAction, QMessageBox


# 训练模型窗口类
class TrainModel(QMainWindow):

    def __init__(self):
        super(TrainModel, self).__init__()

        font = QFont()
        font.setPointSize(8)
        QApplication.setFont(font)

        self.classes = [chr(i) for i in range(ord('A'), ord('Z') + 1) if chr(i) not in ['J', 'Z']]
        self.classes.append('_')
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(max_num_hands=1)
        self.mpDraw = mp.solutions.drawing_utils

        self.setWindowTitle("训练模型")
        self.setWindowIcon(QIcon('icons/main.png'))

        self.train = pd.DataFrame(columns=[str(i) for i in range(25)] + ['label'])  # 存储训练数据的主数据帧
        self.label = '_'  # 现有字母的标签
        self.label_index = 0   # 类列表中当前字母的索引
        self.iterator = 0  # 用于跟踪每个字母处理的帧数的迭代器


        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.image_label = QLabel(self)
        layout = QVBoxLayout(self.central_widget)
        layout.addWidget(self.image_label)

        self.training_layout = QHBoxLayout()
        self.text1 = QLabel('请用手语表示这个字符: ')
        self.text1.setStyleSheet(
            'background-color: white;border:none;color:black;padding: 10px 20px;text-align:center;text-decoration:none;display:inline-block;font-size:16px;')
        self.label_button = QPushButton(str(self.label))
        self.label_button.setStyleSheet(
            'background-color: white;border:none;color:black;padding: 10px 20px;text-align:center;text-decoration:none;display:inline-block;font-size:16px;')
        self.training_layout.addWidget(self.text1)
        self.training_layout.addWidget(self.label_button)
        layout.addLayout(self.training_layout)

        self.interface_layout = QHBoxLayout()
        self.cancel_button = QPushButton('退出')
        self.cancel_button.setStyleSheet(
            'background-color: #f44336;border:none;color:white;padding: 10px 20px;text-align:center;text-decoration:none;display:inline-block;font-size:16px;')
        self.cancel_button.clicked.connect(self.cancel_clicked)
        self.interface_layout.addWidget(self.cancel_button)
        layout.addLayout(self.interface_layout)

        self.cap = cv2.VideoCapture(0)

        self.timer = QTimer()
        self.timer.timeout.connect(self.process_frame)
        self.timer.start(100)

    def normalize(self, entry):
        ref = (entry.loc[0, '0_x'], entry.loc[0, '0_y'])

        hand_points = []
        for i in range(0, 21):
            hand_points.append((entry.loc[0, str(i) + '_x'], entry.loc[0, str(i) + '_y']))

        distances = []
        for point in hand_points:
            distance = math.sqrt((point[0] - ref[0]) ** 2 + (point[1] - ref[1]) ** 2)
            distances.append(distance)

        norm_distances = []
        ref2 = hand_points[12]
        palm_distance = math.sqrt((ref[0] - ref2[0]) ** 2 + (ref[1] - ref2[1]) ** 2)
        finger_distance = math.sqrt(
            (hand_points[8][0] - hand_points[12][0]) ** 2 + (hand_points[8][1] - hand_points[12][1]) ** 2)
        k1 = math.sqrt((hand_points[5][0] - hand_points[9][0]) ** 2 + (hand_points[5][1] - hand_points[9][1]) ** 2)
        k2 = math.sqrt((hand_points[13][0] - hand_points[17][0]) ** 2 + (hand_points[13][1] - hand_points[17][1]) ** 2)
        k3 = math.sqrt((hand_points[9][0] - hand_points[13][0]) ** 2 + (hand_points[9][1] - hand_points[13][1]) ** 2)
        for distance in distances:
            norm_distance = distance / palm_distance
            norm_distances.append(norm_distance)

        norm_distances.append(finger_distance / palm_distance)
        norm_distances.append(k1 / palm_distance)
        norm_distances.append(k2 / palm_distance)
        norm_distances.append(k3 / palm_distance)

        entry_final = pd.DataFrame()

        for i in range(0, 25):
            entry_final.loc[0, str(i)] = norm_distances[i]

        return entry_final

    def train_model(self):
        # 显示基本警告
        alert = QMessageBox()
        alert.setIcon(QMessageBox.Information)
        alert.setText("系统正在训练模型，这可能导致卡顿。")
        alert.setWindowTitle("Alert")
        alert.setStandardButtons(QMessageBox.Ok)
        alert.exec_()

        train = pd.read_csv('data/train.csv')
        X = train.drop('label', axis=1)
        y = train['label']

        from sklearn.preprocessing import OrdinalEncoder
        oe = OrdinalEncoder()
        y = oe.fit_transform(y.to_numpy().reshape(-1, 1))

        # 神经网络
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(64, input_dim=X.shape[1], activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(25, activation='softmax')
        ])


        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        model.fit(X, y, epochs=150)

        model.save('models/model.h5')

        converter = lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        open('models/model.tflite', 'wb').write(tflite_model)

        alert = QMessageBox()
        alert.setIcon(QMessageBox.Information)
        alert.setText("模型训练成功，窗口即将关闭")
        alert.setWindowTitle("Alert")
        alert.setStandardButtons(QMessageBox.Ok)
        alert.exec_()

        sys.exit(0)

    def process_frame(self):
        success, img = self.cap.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)

        if self.iterator < 50 and self.label_index < 25:
            self.label = self.classes[self.label_index]
            self.label_button.setText(str(self.label))
            if results.multi_hand_landmarks:
                train = pd.DataFrame()
                for handVisible in results.multi_hand_landmarks:
                    for id, lm in enumerate(handVisible.landmark):
                        h, w, c = img.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        train.loc[0, str(id) + '_x'] = cx
                        train.loc[0, str(id) + '_y'] = cy

                    self.mpDraw.draw_landmarks(img, handVisible, self.mpHands.HAND_CONNECTIONS)

                # 规范化
                train_final = self.normalize(train)
                train_final['label'] = self.label
                self.train = pd.concat([self.train, train_final], ignore_index=True)

                self.iterator += 1

        else:
            self.iterator = 0
            self.label_index += 1
            self.train.to_csv('data/train.csv', index=False)
            try:    # 当label_index超过类的数量时，训练完成时
                self.label = self.classes[self.label_index]
                alert = QMessageBox()
                alert.setIcon(QMessageBox.Information)
                alert.setText("请表示这个字符: " + str(self.label))
                alert.setWindowTitle("Alert")
                alert.setStandardButtons(QMessageBox.Ok)
                alert.exec_()
            except:
                self.train_model()

        height, width, channel = img.shape
        bytes_per_line = channel * width
        q_image = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.image_label.setPixmap(pixmap)
        self.image_label.setScaledContents(True)

    def cancel_clicked(self):
        self.close()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = TrainModel()
    window.show()
    sys.exit(app.exec_())
