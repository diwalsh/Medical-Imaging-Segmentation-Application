# Import required libraries
import os
import re
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
 
"""
We've downloaded the "train" folder. The original train set images lie inside
the "original" folder.
"""
# Define paths for training dataset and image directory
#TRAIN_CSV = "train.csv"
#ORIG_IMG_DIR = os.path.join("original", "train")
TRAIN_CSV = "data/train_5cases_.csv"
ORIG_IMG_DIR = os.path.join("images", "train")
CASE_FOLDERS = os.listdir(ORIG_IMG_DIR)
 
# Define paths for training and validation image and mask directories
ROOT_DATASET_DIR = "dataset_UWM_GI_Tract_train_valid_5cases" #"dataset_UWM_GI_Tract_train_valid"
ROOT_TRAIN_IMG_DIR = os.path.join(ROOT_DATASET_DIR, "train", "images")
ROOT_TRAIN_MSK_DIR = os.path.join(ROOT_DATASET_DIR, "train", "masks")
ROOT_VALID_IMG_DIR = os.path.join(ROOT_DATASET_DIR, "valid", "images")
ROOT_VALID_MSK_DIR = os.path.join(ROOT_DATASET_DIR, "valid", "masks")
 
# Create directories if not already present
os.makedirs(ROOT_TRAIN_IMG_DIR, exist_ok=True)
os.makedirs(ROOT_TRAIN_MSK_DIR, exist_ok=True)
os.makedirs(ROOT_VALID_IMG_DIR, exist_ok=True)
os.makedirs(ROOT_VALID_MSK_DIR, exist_ok=True)
 
# Define regular expressions to extract case, date, slice number, and image shape from file paths
GET_CASE_AND_DATE = re.compile(r"case[0-9]{1,3}_day[0-9]{1,3}")
GET_SLICE_NUM = re.compile(r"slice_[0-9]{1,4}")
IMG_SHAPE = re.compile(r"_[0-9]{1,3}_[0-9]{1,3}_")
 
# Load the main dataframe from csv file and drop rows with null values, in this way, it only contains relevant images
MAIN_DF = pd.read_csv(TRAIN_CSV).dropna(axis=0)
only_IDS = MAIN_DF["id"].to_numpy()
 
# Define classes for image segmentation
CLASSES = ["large_bowel", "small_bowel", "stomach"]
 
# Create a mapping of class ID to RGB value
color2id = {
    (0, 0, 0): 0,  # background pixel
    (0, 0, 255): 1,  # Blue - Stomach
    (0, 255, 0): 2,  # Green - Small bowel
    (255, 0, 0): 3,  # Red - Large bowel
}
 
# Reverse map from id to color
id2color = {v: k for k, v in color2id.items()}
 
 
# Function to get all relevant image files in a given directory
def get_folder_files(folder_path):
    all_relevant_imgs_in_case = []
    img_ids = []

    # images\train\case11\case11_day0\scans ['slice_0001_360_310_1.50_1.50.png', 'slice_0002_360_310_1.50_1.50.png',  
    #'slice_0003_360_310_1.50_1.50.png', 'slice_0004_360_310_1.50_1.50.png',...]
    for dir, _, files in os.walk(folder_path):
        if not len(files):
            continue
        # goes over the list of image names
        for file_name in files:
            src_file_path = os.path.join(dir, file_name)

            # creates the file_name of the preprocessed images
            case_day = GET_CASE_AND_DATE.search(src_file_path).group()
            slice_id = GET_SLICE_NUM.search(src_file_path).group()
            image_id = case_day + "_" + slice_id

            # checks if the image_id is present in the only_IDS list
            if image_id in only_IDS:
                all_relevant_imgs_in_case.append(src_file_path)
                img_ids.append(image_id)
                
    # It returns the lists all_relevant_imgs_in_case (containing paths to relevant image files) and img_ids (containing corresponding image IDs).
    return all_relevant_imgs_in_case, img_ids
 
 
# Function to decode Run-Length Encoding (RLE) into an image mask
# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_decode(mask_rle, shape):
    """
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
 
    """
    s = np.asarray(mask_rle.split(), dtype=int)
    starts = s[0::2] - 1
    lengths = s[1::2]
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)  # Needed to align to RLE direction
 
 
# Function to load and convert image from a uint16 to uint8 datatype.
def load_img(img_path):
    # reads the image file specified in img_path.
    # cv2.IMREAD_UNCHANGED indicates that the image is loades as-is
    # The result is stored in img and then coverted to float32.
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    # pefroms mix-max normalization on the image array.
    # this normalization ensure that the pixel values are withing the range of 8-bit grayscale or RGB mages
    img = (img - img.min()) / (img.max() - img.min()) * 255.0
    # conversion to unit8 (8-bits)
    img = img.astype(np.uint8)
    # checks if the image is grayscale and, if so, converts it to an RGB image by replicating the single-channel
    # (grayscale) image to three channels (R, G, B) using np.tile()
    img = np.tile(img[..., None], [1, 1, 3])  # gray to rgb
    return img
 
 
# Function to convert RGB image to one-hot encoded grayscale image based on color map.
def rgb_to_onehot_to_gray(rgb_arr, color_map=id2color):
    num_classes = len(color_map)
    # creates a new tuple named shape that contains the height and width dimensions of the image followed by
    # the num_classes value as the third dimension. 
    shape = rgb_arr.shape[:2] + (num_classes,)
    # creates a new NumPy array arr filled with zeros, with the shape determined by shape.
    # This array will store the one-hot encoded representations of the input RGB image.
    arr = np.zeros(shape, dtype=np.float32)

    # conversion looop: iterates over each class in the color_map
    for i, cls in enumerate(color_map):
        # one-hot encoding for each class. It reshapes the input RGB image into a 2D array wher each row 
        # represent a pixel and its RGB values. It then compared each pixel's RGB values to the RGB values
        # of the current class in the 'color_map'. If the RGB values match, it assigns a value of 1 to the
        # corresponding position in 'arr' for that class. Finally, it reshaps the result to match the height
        # and width dimensions of the input image.
        # Basically, this array is a 3D matrix (W, H, 4) with 0s and 1s
        arr[:, :, i] = np.all(rgb_arr.reshape((-1, 3)) == color_map[i], axis=1).reshape(shape[:2])

    # performs an argmax operation along the last axis of the arr array, effectively converting the one-hot encoded
    # representation back to grayscale. The result is an array where each pixel is represented by a single integer
    # corresponding to the class label with the highest value in the one-hot encoded representation.
    # Basically, this is a 2D matrix, with value 0 for background, 1 for stomach, 2 for small bowel, and 3 for large bowel
    return arr.argmax(-1)
 
 
# Function to create and write image-mask pair for each file path in given directories.
def create_and_write_img_msk(file_paths, file_ids, save_img_dir, save_msk_dir, desc=None):
    # iterates over each file_path and file_id pair using zip(file_paths, file_ids), while also displaying a progress bar using tqdm.
    for file_path, file_id in tqdm(zip(file_paths, file_ids), ascii=True, total=len(file_ids), desc=desc, leave=True):
        # loads the image corresponding to the current file_path using the load_img function.
        image = load_img(file_path)

        # retrieves the rows from the DataFrame MAIN_DF where the "id" column matches the current file_id.
        IMG_DF = MAIN_DF[MAIN_DF["id"] == file_id]

        # extracts the height and width of the image from its file path using a regular expression and stores them in img_shape_H_W
        img_shape_H_W = list(map(int, IMG_SHAPE.search(file_path).group()[1:-1].split("_")))[::-1]
        # initializes an array mask_image filled with zeros, with a shape determined by the image dimensions (img_shape_H_W) and the number of classes (len(CLASSES)).
        mask_image = np.zeros(img_shape_H_W + [len(CLASSES)], dtype=np.uint8)

        # iterates over each class label in CLASSES and retrieves the rows from IMG_DF where the "class" column matches the current class label.
        for i, class_label in enumerate(CLASSES):
            class_row = IMG_DF[IMG_DF["class"] == class_label]

            # If there are rows corresponding to the current class label, it retrieves the segmentation mask as run-length
            # encoded string (rle) from the DataFrame and decodes it using the rle_decode function. It then assigns the 
            # decoded mask to the appropriate channel of mask_image.
            if len(class_row):
                rle = class_row.segmentation.squeeze()
                mask_image[..., i] = rle_decode(rle, img_shape_H_W) * 255

        # converts the multi-channel one-hot encoded mask to a grayscale image using the rgb_to_onehot_to_gray function.
        mask_image = rgb_to_onehot_to_gray(mask_image, color_map=id2color)

        # extracts the case and date information from the file path and the file name.
        FILE_CASE_AND_DATE = GET_CASE_AND_DATE.search(file_path).group()
        # splits the file_path into two parts: the directory path and the file name. It returns these two parts as a tuple (directory_path, file_name)
        FILE_NAME = os.path.split(file_path)[-1]

        # constructs new file names for the image and mask files based on the case, date, and original file name.
        # It then creates the destination paths for saving the image and mask files.
        new_name = FILE_CASE_AND_DATE + "_" + FILE_NAME
 
        dst_img_path = os.path.join(save_img_dir, new_name)
        dst_msk_path = os.path.join(save_msk_dir, new_name)

        # writes the image and mask arrays to the corresponding destination paths using cv2.imwrite.
        cv2.imwrite(dst_img_path, image)
        cv2.imwrite(dst_msk_path, mask_image)
 
    return
 
 
if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
 
    # Main script execution: for each folder, split the data into training and validation sets, and create/write image-mask pairs.
    for folder in CASE_FOLDERS:
        all_relevant_imgs_in_case, img_ids = get_folder_files(folder_path=os.path.join(ORIG_IMG_DIR, folder))
        train_files, valid_files, train_img_ids, valid_img_ids = train_test_split(all_relevant_imgs_in_case, img_ids, train_size=0.8, shuffle=True)
        create_and_write_img_msk(train_files, train_img_ids, ROOT_TRAIN_IMG_DIR, ROOT_TRAIN_MSK_DIR, desc=f"Train :: {folder}")
        create_and_write_img_msk(valid_files, valid_img_ids, ROOT_VALID_IMG_DIR, ROOT_VALID_MSK_DIR, desc=f"Valid :: {folder}")
        print()