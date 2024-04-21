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
  
# Function to load and convert image from a uint16 to uint8 datatype.
def load_img(img_path):
    # reads the image file specified in img_path.
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    # pefroms mix-max normalization on the image array.
    img = (img - img.min()) / (img.max() - img.min()) * 255.0
    # conversion to unit8 (8-bits)
    img = img.astype(np.uint8)
 
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
 
import argparse

def main(csv='data/train.csv', root_dir='dataset_UWM_GI_Tract_train_valid'):
    # Process input parameters
    print("CSV:", csv)
    print("Root Dir:", root_dir)
    
    # Set random seed for reproducibility
    np.random.seed(42)

    # Define paths for training dataset and image directory
    TRAIN_CSV = csv
    ORIG_IMG_DIR = os.path.join("images_all", "test")
    CASE_FOLDERS = os.listdir(ORIG_IMG_DIR)


    # Define paths for training and validation image and mask directories
    ROOT_DATASET_DIR = root_dir #+ '_dim' + dimension + '_stride' + str(stride)
    ROOT_TRAIN_DIR_0 = os.path.join(ROOT_DATASET_DIR, "train", "class_0")
    ROOT_TRAIN_DIR_1 = os.path.join(ROOT_DATASET_DIR, "train", "class_1")
    ROOT_VALID_DIR_0 = os.path.join(ROOT_DATASET_DIR, "valid", "class_0")
    ROOT_VALID_DIR_1 = os.path.join(ROOT_DATASET_DIR, "valid", "class_1")
 
    # Create directories if not already present
    os.makedirs(ROOT_TRAIN_DIR_0, exist_ok=True)
    os.makedirs(ROOT_TRAIN_DIR_1, exist_ok=True)
    os.makedirs(ROOT_VALID_DIR_0, exist_ok=True)
    os.makedirs(ROOT_VALID_DIR_1, exist_ok=True)

    # Load the main dataframe from csv file and drop rows with null values, in this way, it only contains relevant images         
    oDF = pd.read_csv(TRAIN_CSV)
    oIDS = oDF["id"].to_numpy()
    mask = (~oDF["segmentation"].isna()).astype(int)
    oDF['classification'] = mask
    
    # Main script execution: for each folder, split the data into training and validation sets, and create/write image-mask pairs.    
    for folder in CASE_FOLDERS:
        all_relevant_imgs_in_case, img_ids = get_folder_files(folder_path=os.path.join(ORIG_IMG_DIR, folder), only_IDS=oIDS)
        train_files, valid_files, train_img_ids, valid_img_ids = train_test_split(all_relevant_imgs_in_case, img_ids, train_size=0.8, random_state=42, shuffle=True)
        create_and_write_img(train_files, train_img_ids, ROOT_TRAIN_DIR_0, ROOT_TRAIN_DIR_1, main_df=oDF, desc=f"Train :: {folder}")
        create_and_write_img(valid_files, valid_img_ids, ROOT_VALID_DIR_0, ROOT_VALID_DIR_1, main_df=oDF, desc=f"Valid :: {folder}")
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Define input parameters   
    parser.add_argument("-csv", type=str, help="path and file name of the csv file with rle data: <path>/<file.csv>")
    parser.add_argument("-dir", type=str, help="Specify the directory where the images will be stored")
    
    args = parser.parse_args()

    # Check if no arguments are provided, then print help
    if not any(vars(args).values()):
        parser.print_help()
    else:
        main(args.csv, args.dir)