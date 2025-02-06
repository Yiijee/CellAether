from PyQt6.QtCore import QThread, pyqtSignal
from CellAether import util

class WorkerThread(QThread):
    progress_updated = pyqtSignal(int)

    def __init__(self, folder_path, process_3D, save_labeled_image, threshold_1, threshold_2):
        super().__init__()
        self.folder_path = folder_path
        self.process_3D = process_3D
        self.save_labeled_image = save_labeled_image
        self.threshold_1 = threshold_1
        self.threshold_2 = threshold_2

    def run(self):
        seg_files, image_files = util.get_files(self.folder_path) # get files for processing
        total_files = len(seg_files)
        for i, (seg_file, image_file) in enumerate(zip(seg_files, image_files)):
            if self.process_3D:
                measurement = util.ROI_measure_intensity(seg_file, image_file)
            else:
                measurement = util.ROI_measure_intensity2D_norm_classify(seg_file, image_file, True, self.save_labeled_image, self.threshold_1, self.threshold_2)
            progress = int((i + 1) / total_files * 100)
            self.progress_updated.emit(progress)
        self.progress_updated.emit(100)  # Signal that progress is complete
