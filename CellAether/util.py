# helper functions for get files, quantify image and save measurements

# import packages

import os
import glob
import numpy as np
import tifffile

def get_files(folder_path):
    """Find all pairs of tiff image and segmentation files in 
    a given folder, return a list of them.

    Args:
        folder_path (str): Any given filder with outputs from cellpose 

    Returns:
        list: segmentation files
        list: images files
    """
    seg_files = glob.glob(os.path.join(folder_path, '*_seg.npy')) # file list of all seg.npy files
    
    existing_seg_files = []
    existing_image_files = []
    # check if image file exists
    for seg_file in seg_files:
        image_file = seg_file[:-8]+".tif"
        if os.path.exists(image_file):
            existing_seg_files.append(seg_file)
            existing_image_files.append(image_file)
    return existing_seg_files, existing_image_files

def ROI_measure(seg_file, image_file):
    """For a given cellpose segmentation in 3D, measure the ROI areas and insensity of pixels
    in each channel

    Args:
        seg_file (string): cellpose _seg.npy output
        image_file (string): tiff image segmented

    Returns:
        measurement: columns are ROI_id, ROI area, chan_1 intensity, chan_2 intensity....
        title: title for each ROI
    """
    # load maskfile
    segmentations = np.load(seg_file,allow_pickle=True).item()
    masks = segmentations["masks"]
    ROI_number = np.max(masks)
    
    # load image 
    image = tifffile.imread(image_file)
    channels = image.shape[1]
    
    # create numpy arry placeholder results
    measurement = np.zeros((ROI_number,channels+2)) # ROI_id, ROI area, chan_1 intensity, chan_2 intensity....
    
    ROI_count = 0
    for z_slice in range(image.shape[0]):
        ROIs = [roi for roi in np.unique(masks[z_slice]) if roi != 0]
        for roi in ROIs:
            roi_mask = masks[z_slice] == roi
            measurement[ROI_count,0] = roi # ROI id
            measurement[ROI_count,1] = np.sum(roi_mask) # ROI area, unit pixel
            for chan in range(channels):
                measurement[ROI_count,chan+2] = np.sum(image[z_slice,chan] * roi_mask) # sum of intensity
            ROI_count += 1
    
    title = ["ROI_ID","ROI_area"] + ["Chan_" + str(i) for i in range(channels)]
    
    return measurement, title

# to be tested

def save_measurement(seg_file, measurement, title):
    results_path = seg_file[:-8]+".csv"
    with open(results_path,'w+') as f:
        f.write(",".join(title)+"\n")
        for entry in measurement:
            f.write(",".join(entry.astype('str'))+"\n")
        
    

if __name__ == "__main__":
    seg_file = "./test/test_2_seg.npy"
    measurement = np.zeros((4,5))
    title = ["a","b","c","d"]
    save_measurement(seg_file,measurement, title)
    
