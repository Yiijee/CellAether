from PyQt6.QtWidgets import QMainWindow, QPushButton, QFileDialog, QProgressBar, QLabel, QCheckBox
import os
from CellAether.workerThread import WorkerThread
from PyQt6.QtWidgets import QLineEdit


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("CellAether: Cellpose ROI Quantification(test)")

        self.button_select_folder = QPushButton("Select Folder", self)
        self.button_select_folder.clicked.connect(self.select_folder)
        self.button_select_folder.setGeometry(100, 50, 200, 50)

        # self.checkbox_kmeans = QCheckBox("Use KMeans Prediction", self)
        # self.checkbox_kmeans.setChecked(True)
        # self.checkbox_kmeans.setGeometry(100, 110, 200, 25)

        self.checkbox_3D = QCheckBox("Quantifying 3D images", self)
        self.checkbox_3D.setChecked(False)
        self.checkbox_3D.setGeometry(100, 110, 200, 25)

        self.checkbox_save_labeled_image = QCheckBox("Save Labeled Image", self)
        self.checkbox_save_labeled_image.setChecked(False)
        self.checkbox_save_labeled_image.setGeometry(100, 140, 200, 25)

        self.label_threshold_1 = QLabel("Threshold 1:", self)
        self.label_threshold_1.setGeometry(100, 170, 100, 25)

        self.input_threshold_1 = QLineEdit(self)
        self.input_threshold_1.setText("1.0")
        self.input_threshold_1.setGeometry(200, 170, 100, 25)

        self.label_threshold_2 = QLabel("Threshold 2:", self)
        self.label_threshold_2.setGeometry(100, 200, 100, 25)

        self.input_threshold_2 = QLineEdit(self)
        self.input_threshold_2.setText("1.0")
        self.input_threshold_2.setGeometry(200, 200, 100, 25)

        self.label_progress = QLabel("Progress:", self)
        self.label_progress.setGeometry(50, 280, 200, 25)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setGeometry(50, 295, 300, 25)

        self.worker_thread = None

    def select_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Directory with Images and _seg.npy Files")
        if folder_path:
            self.start_processing(folder_path)

    def start_processing(self, folder_path):
        self.progress_bar.setValue(0)
        # use_kmeans = self.checkbox_kmeans.isChecked()
        process_3D = self.checkbox_3D.isChecked()
        save_labeled_image = self.checkbox_save_labeled_image.isChecked()
        threshold_1 = float(self.input_threshold_1.text())
        threshold_2 = float(self.input_threshold_2.text())
        self.worker_thread = WorkerThread(folder_path, process_3D, save_labeled_image, threshold_1, threshold_2)
        self.worker_thread.progress_updated.connect(self.update_progress)
        self.worker_thread.start()

    def update_progress(self, progress):
        self.progress_bar.setValue(progress)
        if progress == 100:
            folder_name = os.path.basename(self.worker_thread.folder_path)
            self.label_progress.setText(f"'{folder_name}' is done!")

