from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget, QPushButton, \
    QHBoxLayout, QMessageBox, QFrame, QScrollArea, QGridLayout
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QFont
import sys


# 帮助类的窗口
class Help(QMainWindow):

    def __init__(self):
        super(Help, self).__init__()
        self.resize(800,400)
        self.setWindowTitle('如何使用此应用程序？')

        self.icon = QIcon('../icons/info.png')
        self.setWindowIcon(self.icon)

        font = QFont()
        font.setPointSize(8)
        QApplication.setFont(font)

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        layout = QGridLayout(self.central_widget)

        self.nav = QVBoxLayout()

        self.option1 = QPushButton('如何使用这个软件进行翻译？')
        self.option1.setStyleSheet(
            'background-color: white;border:none;color:black;padding: 10px 20px;text-align:center;text-decoration:none;display:inline-block;font-size:16px;')
        self.option1.clicked.connect(self.option1_clicked)
        self.option2 = QPushButton('H如何使模型更加准确？')
        self.option2.setStyleSheet(
            'background-color: white;border:none;color:black;padding: 10px 20px;text-align:center;text-decoration:none;display:inline-block;font-size:16px;')
        self.option2.clicked.connect(self.option2_clicked)
        self.option3 = QPushButton('如何重新训练模型？')
        self.option3.setStyleSheet(
            'background-color: white;border:none;color:black;padding: 10px 20px;text-align:center;text-decoration:none;display:inline-block;font-size:16px;')
        self.option3.clicked.connect(self.option3_clicked)

        self.nav.addWidget(self.option1)
        self.nav.addWidget(self.option2)
        self.nav.addWidget(self.option3)
        self.nav.addStretch(1)

        self.main_display = QWidget()
        self.main_layout = QVBoxLayout(self.main_display)
        self.main_layout.setContentsMargins(50, 10, 50, 10)

        self.display = QLabel()
        self.display.setWordWrap(True)
        self.display.adjustSize()
        self.display.setText(
            '该应用程序将访问你的网络摄像头，并将手语翻译成文本.如果检测到，它将在文本框中显示翻译后的文本')
        self.main_layout.addWidget(self.display)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        scroll_area.setWidget(self.main_display)

        seperator = QFrame()
        seperator.setFrameShape(QFrame.VLine)
        seperator.setFrameShadow(QFrame.Sunken)
        seperator.setLineWidth(3)
        seperator.setStyleSheet("color: rgb(0, 0, 0);")

        layout.addLayout(self.nav, 0, 0, 1, 1)
        layout.addWidget(seperator, 0, 1, 1, 1)
        layout.addWidget(scroll_area, 0, 2, 1, 4)

        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 2)

        self.option1_clicked()

    def option1_clicked(self):
        self.display.setText(
            '该应用程序将访问你的网络摄像头，并将手语翻译成文本.如果检测到，它将在文本框中显示翻译后的文本')

        self.reset_option_clicks()

        self.option1.setEnabled(False)
        self.option1.setStyleSheet(
            'background-color: gray;border:none;color:black;padding: 10px 20px;text-align:center;text-decoration:none;display:inline-block;font-size:16px;')
        self.option2.setEnabled(True)
        self.option3.setEnabled(True)

    def option2_clicked(self):
        self.display.setText(
            '如果响应准确，请对文本字段下方的提示做出“是”响应。在模型选项下，单击重新训练模型以刷新模型，使用新数据。该模型将使用新数据进行再培训，并将更加准确。')

        self.reset_option_clicks()

        self.option1.setEnabled(True)
        self.option2.setEnabled(False)
        self.option2.setStyleSheet(
            'background-color: gray;border:none;color:black;padding: 10px 20px;text-align:center;text-decoration:none;display:inline-block;font-size:16px;')
        self.option3.setEnabled(True)

    def option3_clicked(self):
        self.display.setText(
            '在模型选项下，单击更新模型，然后从头开始训练模型。这需要一段时间。')

        self.reset_option_clicks()

        self.option1.setEnabled(True)
        self.option2.setEnabled(True)
        self.option3.setEnabled(False)
        self.option3.setStyleSheet(
            'background-color: gray;border:none;color:black;padding: 10px 20px;text-align:center;text-decoration:none;display:inline-block;font-size:16px;')

    # 重置选项上的单击
    def reset_option_clicks(self):
        self.option1.setStyleSheet(
            'background-color: white;border:none;color:black;padding: 10px 20px;text-align:center;text-decoration:none;display:inline-block;font-size:16px;')
        self.option2.setStyleSheet(
            'background-color: white;border:none;color:black;padding: 10px 20px;text-align:center;text-decoration:none;display:inline-block;font-size:16px;')
        self.option3.setStyleSheet(
            'background-color: white;border:none;color:black;padding: 10px 20px;text-align:center;text-decoration:none;display:inline-block;font-size:16px;')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Help()
    window.show()
    sys.exit(app.exec_())
