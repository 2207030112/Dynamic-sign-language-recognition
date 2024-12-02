from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
import sys

# 关于应用程序窗口的类
class aboutApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.resize(800, 400)
        self.setWindowTitle('About this App')

        self.icon = QIcon('../icons/info.png')
        self.setWindowIcon(self.icon)

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        layout = QVBoxLayout(self.central_widget)

        self.display1 = QLabel()
        self.display1.setWordWrap(True)
        self.display1.setContentsMargins(50, 10, 50, 10)
        self.display1.adjustSize()
        self.display1.setAlignment(Qt.AlignCenter)
        self.display1.setText('这个应用程序是一个手语翻译的原型，由第十小组开发.')

        self.display2 = QLabel()
        self.display2.setContentsMargins(50, 10, 50, 10)
        self.display2.setAlignment(Qt.AlignCenter)
        self.display2.setOpenExternalLinks(True)
        self.display2.setText('此应用将与另一个程序之间传输数据，达到手语控制肢体动作的效果')

        self.display3 = QLabel()
        self.display3.setContentsMargins(50, 10, 50, 10)
        self.display3.setAlignment(Qt.AlignCenter)
        self.display3.setOpenExternalLinks(True)
        self.display3.setText('感谢您的使用.')

        layout.addWidget(self.display1)
        layout.addWidget(self.display2)
        layout.addWidget(self.display3)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = aboutApp()
    window.show()
    sys.exit(app.exec_())