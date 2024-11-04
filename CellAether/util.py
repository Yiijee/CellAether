# helper functions for get files, quantify image and save measurements

# import packages

import os
import glob
import numpy as np
import tifffile
import pandas as pd

# IO functions
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

# Currently not used
def ROI_classify(seg_file, measurement, features_col, model_path = None): 
    """Classify ROIs based on the features extracted from ROI_measure

    Args:
        measurement (pandasDataframe): output from ROI_measure
        features (list): list of features index to be used for classification
        model_path (str, optional): path to the model. Defaults to "kmeans.pkl".

    Returns:
        measurement: meansurement with classification labels
    """
    from sklearn.cluster import KMeans
    import pickle
    import pkg_resources
    if model_path is None:
        model_path = pkg_resources.resource_filename('CellAether', 'kmeans.pkl')
    # load model
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    # select features
    features = measurement.iloc[:,features_col]
    # classify
    classification = model.predict(features)
    measurement["classification"] = classification
    results_path = seg_file[:-8]+".csv"
    measurement.to_csv(results_path)
    return measurement


def label_ROIs(seg_file, image_file, measurement):
    """Label ROIs in the image with the classification label

    Args:
        seg_file (string): cellpose _seg.npy output
        image_file (string): tiff image segmented
        measurement (pandasDataframe): output from ROI_measure

    Returns:
        None
    """
    # load maskfile
    segmentations = np.load(seg_file,allow_pickle=True).item()
    masks = segmentations["masks"]
    # class number 
    class_number = len(np.unique(measurement["classification"]))
    # load image 
    image = tifffile.imread(image_file)
    channels = image.shape[1]
    image_labeled = np.zeros((image.shape[0],image.shape[1]+class_number,*image.shape[2:]))
    image_labeled[:,:channels] = image
    # create numpy arry placeholder results
    for roi_id, classification in measurement.loc[:,["ROI_ID","classification"]].values:
        for z_slice in range(image.shape[0]):
            roi_mask = masks[z_slice] == roi_id 
            image_labeled[z_slice, int(channels+classification), roi_mask] = 100
    image_labeled_unit8 = image_labeled.astype(np.uint8)
    tifffile.imwrite(image_file[:-4]+"_labelled.tif", image_labeled_unit8,imagej=True)


def ROI_measure_intensity(seg_file, image_file, save_reults = True):
    """For a given cellpose segmentation in 3D, measure the ROI areas and insensity of pixels
    in each channel

    Args:
        seg_file (string): cellpose _seg.npy output
        image_file (string): tiff image segmented
        save_reults (bool, optional): save the results to a csv file. Defaults to True.

    Returns:
        measurement: columns are ROI_id, ROI area, chan_1 intensity, chan_2 intensity....
    """
    # load maskfile
    segmentations = np.load(seg_file,allow_pickle=True).item()
    masks = segmentations["masks"]
    ROI_number = np.max(masks)
    
    # load image 
    image = tifffile.imread(image_file)
    channels = image.shape[1]
    
    # create numpy arry placeholder results
    measurement = np.zeros((ROI_number,channels*3+2+1)) # ROI_id, ROI area, chan_1 intensity, chan_2 intensity....
    title = ["ROI_ID","ROI_area"]
    for ch in range(channels):
        title.append(f"Chan_{ch}_mean")
        title.append(f"Chan_{ch}_std")
        title.append(f"Chan_{ch}_max")
    title.append("classification")
    mean_intensities = [np.mean(image[:,chan]) for chan in range(channels)]
    std_intensities = [np.std(image[:,chan]) for chan in range(channels)]
    for z_slice in range(image.shape[0]):
        print(z_slice)
        ROIs = [roi for roi in np.unique(masks[z_slice]) if roi != 0]
        for roi in ROIs:
            ROI_count = roi - 1
            roi_mask = masks[z_slice] == roi
            measurement[ROI_count,0] = roi # ROI id
            measurement[ROI_count,1] = np.sum(roi_mask) # ROI area, unit pixel

            for chan in range(channels):
                # Perform z-score normalization
                channel_data = image[z_slice, chan]
                mean_intensity = mean_intensities[chan]
                std_intensity = std_intensities[chan]
                if std_intensity != 0:
                    normalized_intensity = (channel_data - mean_intensity) / std_intensity
                else:
                    normalized_intensity = channel_data - mean_intensity
                measurement[ROI_count, chan*3+2] = np.mean(normalized_intensity* roi_mask)
                measurement[ROI_count, chan*3+3] = np.std(normalized_intensity* roi_mask)
                measurement[ROI_count, chan*3+4] = np.max(normalized_intensity* roi_mask)
            ROI_count += 1
    measurement = pd.DataFrame(measurement, columns=title)
    if save_reults:
        results_path = seg_file[:-8]+"measurments.csv"
        measurement.to_csv(results_path)
    return measurement

def ROI_measure_HCR(seg_file, image_file, HCR_channels=[2,3],save_reults = True):
    """For a given cellpose segmentation in 3D, measure the ROI areas and HCR counts and intensity
    in each channel

    Args:
        seg_file (str): cellpose _seg.npy output
        image_file (_type_): tiff image segmented
        HCR_channels (list, optional): Image channels for HCR. Defaults to [2,3].
        save_reults (bool, optional): save the results to a csv file. Defaults to True.
    Returns:
        measurement: A long dataframe with columns: ROI_ID,Channel, HCR_intensity, HCR_prob
    """
    from spotiflow.model import Spotiflow
    # load image 
    image = tifffile.imread(image_file)
    channels = image.shape[1]

    # load maskfile
    segmentations = np.load(seg_file,allow_pickle=True).item()
    masks = segmentations["masks"]
    ROI_number = np.max(masks)

    # placeholder for results
    HCR_menasurement = {
        "ROI_ID":[],
        "Channel":[],
        "HCR_intensity":[],
        "HCR_prob":[]
    }
    
    # Load a pretrained model
    model = Spotiflow.from_pretrained("general")

    for z_slice in range(image.shape[0]):
        print(z_slice)
        mask_slice = masks[z_slice]
        for ch in HCR_channels:
            img = image[z_slice, ch, :, :]
            # Predict
            points, details = model.predict(img)
            # Iterate through each point
            for i,point in enumerate(points):
                x, y = int(point[0]), int(point[1])
                # Check which ROI the point belongs to
                roi = mask_slice[x, y]
                if roi != 0:
                    HCR_menasurement["ROI_ID"].append(roi)
                    HCR_menasurement["Channel"].append(ch)
                    HCR_menasurement["HCR_intensity"].append(details.intens[i,0])
                    HCR_menasurement["HCR_prob"].append(details.prob[i])
    HCR_menasurement = pd.DataFrame(HCR_menasurement)
    if save_reults:
        results_path = seg_file[:-8]+"HCR_measuements.csv"
        pd.to_csv(results_path)
    return HCR_menasurement
    

def extractROIs(seg_file):
    # load maskfile
    segmentations = np.load(seg_file,allow_pickle=True).item()
    masks = segmentations["masks"]
    ROI_number = np.max(masks)
    # Create a list to store cropped ROI images
    cropped_ROIs = []
    ROIs = []
    ROI_z = []

    for z_slice in range(masks.shape[0]):
        ROIs = [roi for roi in np.unique(masks[z_slice]) if roi != 0]

        for roi in ROIs:
            roi_mask = masks[z_slice] == roi
            # Find the bounding box of the ROI
            y = np.any(roi_mask, axis=1)
            x = np.any(roi_mask, axis=0)
            y_min, y_max = np.where(y)[0][[0, -1]]
            x_min, x_max = np.where(x)[0][[0, -1]]
            cropped_ROIs.append((z_slice, y_min, y_max, x_min, x_max))
            ROIs.append(roi_mask)
            ROI_z.append
        
    return ROIs

def cropImage(image, roi):
        maksed_image = image
        # maksed_image = np.zeros_like(image)
        # for ch in range(image.shape[0]):
            # maksed_image[ch] = image[ch] * roi

        y = np.any(roi, axis=1)
        x = np.any(roi, axis=0)
        y_min, y_max = np.where(y)[0][[0, -1]]
        x_min, x_max = np.where(x)[0][[0, -1]]
        cropped_image = maksed_image[:, y_min-20:y_max+20, x_min-20:x_max+20]
        cropped_image = np.concatenate([cropped_image, roi[np.newaxis, y_min-20:y_max+20, x_min-20:x_max+20]], axis=0)

        return cropped_image

    # # Load the image
    # image = tifffile.imread(image_file)
    # channels = image.shape[1]

    # # Randomly select n ROIs from cropped_ROIs
    # selected_ROIs = np.random.choice(len(cropped_ROIs), n, replace=False)
    # # Create a directory to save cropped images
    # cropped_image_dir = os.path.join(os.path.dirname(image_file), "cropped_image")
    # os.makedirs(cropped_image_dir, exist_ok=True)
    
    # cropped_images = []
    # for idx in selected_ROIs:
    #     z_slice, y_min, y_max, x_min, x_max = cropped_ROIs[idx]
    #     image_slice = image[z_slice] * ROIs[idx]
    #     image[z_slice, :, y_min:y_max, x_min:x_max]
    #     cropped_image = image[z_slice, :, y_min:y_max, x_min:x_max]
    #     cropped_images.append(cropped_image)
    #     cropped_image_path = os.path.join(cropped_image_dir, f"cropped_image_ROI_{idx}.tif")
    #     tifffile.imwrite(cropped_image_path, cropped_image)


# to be tested

# def save_measurement(seg_file, measurement, title):
#     results_path = seg_file[:-8]+".csv"
#     with open(results_path,'w+') as f:
#         f.write(",".join(title)+"\n")
#         for entry in measurement:
#             f.write(",".join(entry.astype('str'))+"\n")
        
    
