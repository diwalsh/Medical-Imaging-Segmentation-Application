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

# Define regular expressions to extract case, date, slice number, and image shape from file paths
GET_CASE_AND_DATE = re.compile(r"case[0-9]{1,3}_day[0-9]{1,3}")
GET_SLICE_NUM = re.compile(r"slice_[0-9]{1,4}")
IMG_SHAPE = re.compile(r"_[0-9]{1,3}_[0-9]{1,3}_")

# Function to get all relevant image files in a given directory
def get_folder_files(folder_path, only_IDS):
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

# Function to get all relevant image files in a given directory
def get_folder_files_2p5d(folder_path, only_IDS, stride=1):
    all_relevant_imgs_in_case = []
    img_ids = []

    # images\train\case11\case11_day0\scans ['slice_0001_360_310_1.50_1.50.png', 'slice_0002_360_310_1.50_1.50.png',  
    #'slice_0003_360_310_1.50_1.50.png', 'slice_0004_360_310_1.50_1.50.png',...]
    for dir, _, files in os.walk(folder_path):
        if not len(files):
            continue
        # goes over the list of image names and creates 2.5d image
        L = len(files)
        for idx, _ in enumerate(files):            
            src_file_path = os.path.join(dir, files[idx])
            src_file_path_p1 = os.path.join(dir, files[min(idx + 1*stride, L-1)])
            src_file_path_p2 = os.path.join(dir, files[min(idx + 2*stride, L-1)])
            src_file_paths = [src_file_path, src_file_path_p1, src_file_path_p2]
            
            # creates the file_name of the preprocessed images
            case_day = GET_CASE_AND_DATE.search(src_file_path).group()
            slice_id = GET_SLICE_NUM.search(src_file_path).group()
            image_id = case_day + "_" + slice_id

            # checks if the image_id is present in the only_IDS list
            if image_id in only_IDS:
                all_relevant_imgs_in_case.append(src_file_paths)
                img_ids.append(image_id)

    # It returns the lists all_relevant_imgs_in_case (containing paths to relevant image files) and img_ids (containing corresponding image IDs).
    return all_relevant_imgs_in_case, img_ids

# Function to load and convert image from a uint16 to uint8 datatype.
def load_img(img_path):
    # reads the image file specified in img_path.
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    # pefroms mix-max normalization on the image array.
    img = (img - img.min()) / (img.max() - img.min()) * 255.0
    # conversion to unit8 (8-bits)
    img = img.astype(np.uint8)
    img = np.tile(img[..., None], [1, 1, 3])  # gray to rgb 
    return img
         

# Function to load three adjacent images and store them in an RGB image format
# img_paths is a list of image paths: [path/img_i-1.png, path/img_i.png, path/img_i+1.png]
def load_img_2p5d(img_paths):
    no_images = len(img_paths)
    img_shape = list(map(int, IMG_SHAPE.search(img_paths[0]).group()[1:-1].split("_")))[::-1]
    img_shape.append(no_images)
    img = np.zeros(tuple(img_shape))
    for idx, img_path in enumerate(img_paths):
        # reads the image file specified in img_path.
        ch = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        # performs mix-max normalization on the image array.
        ch = (ch - ch.min()) / (ch.max() - ch.min()) * 255.0
        # conversion to unit8 (8-bits)
        ch = ch.astype(np.uint8)
        img[:,:,idx] = ch
        if idx == 2:
            break
    
    return img

# Function to create and write image-mask pair for each file path in given directories.
def create_and_write_img(file_paths, file_ids, save_dir_0, save_dir_1, main_df, desc=None):
    # iterates over each file_path and file_id pair using zip(file_paths, file_ids), while also displaying a progress bar using tqdm.
    for file_path, file_id in tqdm(zip(file_paths, file_ids), ascii=True, total=len(file_ids), desc=desc, leave=True):
        # loads the image corresponding to the current file_path using the load_img function.
        image = load_img(file_path)

        # retrieves the rows from the DataFrame MAIN_DF where the "id" column matches the current file_id.
        IMG_DF = main_df[main_df["id"] == file_id]
        # detects if the image contains the organs of interest (stomach, small bowel, larg bowel)
        all_zeros = (IMG_DF["classification"] == 0).all()
        
        # extracts the case and date information from the file path and the file name.
        FILE_CASE_AND_DATE = GET_CASE_AND_DATE.search(file_path).group()
        FILE_SLICE_NUMBER = GET_SLICE_NUM.search(file_path).group()
        
        # splits the file_path into two parts: the directory path and the file name. It returns these two parts as a tuple (directory_path, file_name)
        FILE_NAME = os.path.split(file_path)[-1]

        # constructs new file names for the image and mask files based on the case, date, and original file name.
        # It then creates the destination paths for saving the image and mask files.
        new_name = FILE_CASE_AND_DATE + "_" + FILE_SLICE_NUMBER + ".png"      
        #FILE_NAME

        if all_zeros:
            dst_img_path = os.path.join(save_dir_0, new_name)
        else:
            dst_img_path = os.path.join(save_dir_1, new_name)
            
        # writes the image and mask arrays to the corresponding destination paths using cv2.imwrite.
        cv2.imwrite(dst_img_path, image)
        
    return

# Function to create and write image-mask pair for each file path in given directories.
def create_and_write_img_2p5d(file_paths, file_ids, save_dir_0, save_dir_1, main_df, desc=None):
    # iterates over each file_path and file_id pair using zip(file_paths, file_ids), while also displaying a progress bar using tqdm.
    for file_path, file_id in tqdm(zip(file_paths, file_ids), ascii=True, total=len(file_ids), desc=desc, leave=True):
        # loads the image corresponding to the current file_path using the load_img function.
        image = load_img_2p5d(file_path)

        # retrieves the rows from the DataFrame MAIN_DF where the "id" column matches the current file_id.
        IMG_DF = main_df[main_df["id"] == file_id]
        # detects if the image contains the organs of interest (stomach, small bowel, larg bowel)
        all_zeros = (IMG_DF["classification"] == 0).all()
        
        # extracts the case and date information from the file path and the file name.
        FILE_CASE_AND_DATE = GET_CASE_AND_DATE.search(file_path[0]).group()
        FILE_SLICE_NUMBER = GET_SLICE_NUM.search(file_path[0]).group()
        
        # splits the file_path into two parts: the directory path and the file name. It returns these two parts as a tuple (directory_path, file_name)
        FILE_NAME = os.path.split(file_path[0])[-1]

        # constructs new file names for the image and mask files based on the case, date, and original file name.
        # It then creates the destination paths for saving the image and mask files.
        #new_name = FILE_CASE_AND_DATE + "_" + FILE_SLICE_NUMBER + ".png"
        new_name = FILE_CASE_AND_DATE + "_" + FILE_NAME
        #FILE_NAME

        if all_zeros:
            dst_img_path = os.path.join(save_dir_0, new_name)
        else:
            dst_img_path = os.path.join(save_dir_1, new_name)
            
        # writes the image and mask arrays to the corresponding destination paths using cv2.imwrite.
        cv2.imwrite(dst_img_path, image)
        
    return

import argparse

def main(dimension, stride, csv_train, csv_test, input_dir, output_dir):
    # Process input parameters
    print("Dimension:", dimension)
    print("Stride:", stride)
    print("CSV Train:", csv_train)
    print("CSV Test:", csv_test)
    print("Input Dir:", input_dir)
    print("Output Dir:", output_dir)
    
    # Set random seed for reproducibility
    np.random.seed(42)

    # Define paths for training dataset and image directory
    TRAIN_CSV = csv_train
    TEST_CSV = csv_test
    ORIG_IMG_DIR_train = os.path.join(input_dir, "train")
    ORIG_IMG_DIR_test = os.path.join(input_dir, "test")
    CASE_FOLDERS_train = os.listdir(ORIG_IMG_DIR_train)
    CASE_FOLDERS_test = os.listdir(ORIG_IMG_DIR_test)

    # Define paths for training and validation image and mask directories
    ROOT_DATASET_DIR = output_dir
    ROOT_TRAIN_DIR_0 = os.path.join(ROOT_DATASET_DIR, "train", "0")
    ROOT_TRAIN_DIR_1 = os.path.join(ROOT_DATASET_DIR, "train", "1")
    ROOT_TEST_DIR_0 = os.path.join(ROOT_DATASET_DIR, "test", "0")
    ROOT_TEST_DIR_1 = os.path.join(ROOT_DATASET_DIR, "test", "1")
     
    # Create directories if not already present
    os.makedirs(ROOT_TRAIN_DIR_0, exist_ok=True)
    os.makedirs(ROOT_TRAIN_DIR_1, exist_ok=True)
    os.makedirs(ROOT_TEST_DIR_0, exist_ok=True)
    os.makedirs(ROOT_TEST_DIR_1, exist_ok=True)

    # Load the main dataframe from csv file and drop rows with null values, in this way, it only contains relevant images         
    oDF_train = pd.read_csv(TRAIN_CSV)
    oIDS_train = oDF_train["id"].to_numpy()
    mask_train = (~oDF_train["segmentation"].isna()).astype(int)
    oDF_train['classification'] = mask_train

    oDF_test = pd.read_csv(TEST_CSV)
    oIDS_test = oDF_test["id"].to_numpy()
    mask_test = (~oDF_test["segmentation"].isna()).astype(int)
    oDF_test['classification'] = mask_test
    
    # Main script execution: for each folder, split the data into training and validation sets, and create/write image-mask pairs.    
    for folder in CASE_FOLDERS_train:
        if dimension == '2d':
            files, ids = get_folder_files(folder_path=os.path.join(ORIG_IMG_DIR_train, folder), only_IDS=oIDS_train)
            create_and_write_img(files, ids, ROOT_TRAIN_DIR_0, ROOT_TRAIN_DIR_1, main_df=oDF_train, desc=f"Train :: {folder}")
        else:
            files, ids = get_folder_files_2p5d(folder_path=os.path.join(ORIG_IMG_DIR_train, folder), only_IDS=oIDS_train, stride=stride)
            create_and_write_img_2p5d(files, ids, ROOT_TRAIN_DIR_0, ROOT_TRAIN_DIR_1, main_df=oDF_train, desc=f"Train :: {folder}")

    for folder in CASE_FOLDERS_test:
        if dimension == '2d':
            files, ids = get_folder_files(folder_path=os.path.join(ORIG_IMG_DIR_test, folder), only_IDS=oIDS_test)
            create_and_write_img(files, ids, ROOT_TEST_DIR_0, ROOT_TEST_DIR_1, main_df=oDF_test, desc=f"Test :: {folder}")
        else:
            files, ids = get_folder_files_2p5d(folder_path=os.path.join(ORIG_IMG_DIR_test, folder), only_IDS=oIDS_test, stride=stride)
            create_and_write_img_2p5d(files, ids, ROOT_TEST_DIR_0, ROOT_TEST_DIR_1, main_df=oDF_test, desc=f"Test :: {folder}")

    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Define input parameters   
    parser.add_argument("-dimension", choices=['2d', '2.5d'], default='2d', help="Choose either '2d' or '2.5d'")
    parser.add_argument("-stride", type=int, default=1, help="Specify the stride as an integer (default 1) for 2.5d")
    parser.add_argument("-csv_train", type=str, default="data/train_train.csv", help="path and file name of the csv train file")
    parser.add_argument("-csv_test", type=str, default="data/train_test.csv", help="path and file name of the csv test file")
    parser.add_argument("-input_dir", type=str, default='images_all', help="Specify the directory where the input images reside (default 'images')")
    parser.add_argument("-output_dir", type=str, default='output', help="Specify the directory where the images will be stored (default 'output')")
    
    args = parser.parse_args()

    # Check if no arguments are provided, then print help
    if not vars(args):
        parser.print_help()
    else:
        # Call the main function with the parsed arguments
        main(args.dimension, args.stride, args.csv_train, args.csv_test, args.input_dir, args.output_dir)