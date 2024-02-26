from PyQt6.QtCore import QThread, pyqtSignal
from CellAether import util

class WorkerThread(QThread):
    progress_updated = pyqtSignal(int)

    def __init__(self, folder_path):
        super().__init__()
        self.folder_path = folder_path

    def run(self):
        seg_files, image_files = util.get_files(self.folder_path) # get files for processing
        total_files = len(seg_files)
        for i, seg_file in enumerate(seg_files):
            image_file = image_files[i]
            measurement, title = util.ROI_measure(seg_file, image_file)
            util.save_measurement(seg_file, measurement, title)
            progress = int((i + 1) / total_files * 100)
            self.progress_updated.emit(progress)
        self.progress_updated.emit(100)  # Signal that progress is complete
