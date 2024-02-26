from PyQt6.QtWidgets import QMainWindow, QPushButton, QFileDialog, QProgressBar, QLabel
import os
from CellAether.workerThread import WorkerThread


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("CellAether: Cellpose ROI Quantification")

        self.button_select_folder = QPushButton("Select Folder", self)
        self.button_select_folder.clicked.connect(self.select_folder)
        self.button_select_folder.setGeometry(100, 50, 200, 50)

        self.label_progress = QLabel("Progress:", self)
        self.label_progress.setGeometry(50, 150, 200, 25)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setGeometry(50, 175, 300, 25)

        self.worker_thread = None

    def select_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Directory with Images and _seg.npy Files")
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
            self.label_progress.setText(f"'{folder_name}' is done!")

