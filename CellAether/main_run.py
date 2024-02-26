from PyQt6.QtWidgets import QApplication
from CellAether.mainWindow import MainWindow
import sys
def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.setGeometry(100, 100, 400, 210)
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()