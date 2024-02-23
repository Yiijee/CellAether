import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QProgressBar, QLabel
from PyQt6.QtCore import QThread, pyqtSignal
import time
import os

class WorkerThread(QThread):
    progress_updated = pyqtSignal(int)

    def __init__(self, folder_path):
        super().__init__()
        self.folder_path = folder_path

    def run(self):
        total_files = len(os.listdir(self.folder_path))
        for i, file in enumerate(os.listdir(self.folder_path)):
            # Simulating some work
            time.sleep(0.1)
            progress = int((i + 1) / total_files * 100)
            self.progress_updated.emit(progress)
        self.progress_updated.emit(100)  # Signal that progress is complete


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("File Processing App")

        self.button_select_folder = QPushButton("Select Folder", self)
        self.button_select_folder.clicked.connect(self.select_folder)
        self.button_select_folder.setGeometry(50, 50, 200, 50)

        self.label_progress = QLabel("Progress:", self)
        self.label_progress.setGeometry(50, 100, 100, 25)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setGeometry(150, 100, 200, 25)

        self.worker_thread = None

    def select_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Directory")
        if folder_path:
            self.start_processing(folder_path)

    def start_processing(self, folder_path):
        self.progress_bar.setValue(0)
        self.worker_thread = WorkerThread(folder_path)
        self.worker_thread.progress_updated.connect(self.update_progress)
        self.worker_thread.start()

    def update_progress(self, progress):
        self.progress_bar.setValue(progress)
        if progress == 100:
            folder_name = os.path.basename(self.worker_thread.folder_path)
            self.label_progress.setText(f"{folder_name} is done!")


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.setGeometry(100, 100, 400, 200)
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
