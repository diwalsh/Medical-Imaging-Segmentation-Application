import os
import cv2
import numpy as np
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import lightning.pytorch as pl
from torchvision import transforms
from transformers import SegformerForSemanticSegmentation

from keras.models import load_model
import albumentations as A
from albumentations.pytorch import ToTensorV2


# class for dataset configuration
@dataclass(frozen=True)
class DatasetConfig:
    NUM_CLASSES:   int = 4 # including background!
    IMAGE_SIZE: tuple[int,int] = (288, 288) # W, H
    MEAN: tuple = (0.485, 0.456, 0.406)
    STD:  tuple = (0.229, 0.224, 0.225)
    MEAN_CLF: float = 0.136
    STD_CLF: float = 0.178
    BACKGROUND_CLS_ID: int = 0
    THR: float = 0.07652711868286133
   
# class for inference configuration   
@dataclass
class InferenceConfig:
    BATCH_SIZE:  int = 12 # can increase the batch size to find faster optimization 
    NUM_BATCHES: int = 2
    
# mapping of class ID to RGB value. (earthy pink tones now)
id2color = {
    0: (0, 0, 0),    # background pixel
    1: (191, 187, 249),  # Stomach
    2: (162, 128, 254), # Small Bowel
    3: (100, 82, 183),  # large Bowel
}

# lightning module class for segmentation model
class MedicalSegmentationModel(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        num_classes: int = 10,
        init_lr: float = 0.001,
        optimizer_name: str = "Adam",
        weight_decay: float = 1e-4,
        use_scheduler: bool = False,
        scheduler_name: str = "multistep_lr",
        num_epochs: int = 1,
        verbose: int = 0
    ):
        super().__init__()

        # Save the arguments as hyperparameters.
        self.save_hyperparameters()

        # Loading model using the function defined above.
        self.model = get_model(model_name=self.hparams.model_name, num_classes=self.hparams.num_classes)

    def forward(self, data):
        outputs = self.model(pixel_values=data, return_dict=True)
        upsampled_logits = F.interpolate(outputs["logits"], size=data.shape[-2:], mode="bilinear", align_corners=False)
        return upsampled_logits
    

# converting numpy array to rgb values
def num_to_rgb(num_arr, color_map=id2color):
    single_layer = np.squeeze(num_arr)
    output = np.zeros(num_arr.shape[:2] + (3,))

    for k in color_map.keys():
        output[single_layer == k] = color_map[k]

    # return a floating point array in range [0.0, 1.0]
    return np.float32(output) / 255.0


# Function to overlay a segmentation map on top of an RGB image.
def image_overlay(image, segmented_image):
    alpha = 1.0  # Transparency for the original image.
    beta = 0.99  # Transparency for the segmentation map.
    gamma = 0.0  # Scalar added to each sum.

    # RBG to BGR for both, overlay transparent masks over CT scan, back to RGB
    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # Set to zero the segmented pixels in the images to ensure opacity after overlay
    image = image * np.where(segmented_image == 0, 1, 0).astype(np.uint8)
    # apply mask overlay
    image = cv2.addWeighted(image, alpha, segmented_image, beta, gamma, image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return np.clip(image, 0.0, 1.0)


# load and resize the image to the specified size.
# The interpolation method used is nearest-neighbor for the segmentation model.
def load_file_nearest(size, file_path, depth=0):
    file = cv2.imread(file_path, depth)
    if depth == cv2.IMREAD_COLOR:
        file = file[:, :, ::-1]
    return cv2.resize(file, (size), interpolation=cv2.INTER_NEAREST)


# load and resize the image to the specified size.
# The interpolation method used is linear for the classification model.
def load_file_linear(size, file_path, depth=0):
    file = cv2.imread(file_path, depth)
    if depth == cv2.IMREAD_COLOR:
        file = file[:, :, ::-1]
    return cv2.resize(file, (size), interpolation=cv2.INTER_LINEAR)


# Classification: normalize image using the specified mean and standard deviation values
def normalize_classif(image, mean, std):
    image = image / 255.0
    image = (image - mean) / std
    return image


# Segmentation: compose mean-std normalization and to-tensor pipeline
def setup_transforms(mean, std):
    transforms = []
    transforms.extend([
        A.Normalize(mean=mean, std=std, always_apply=True),
        ToTensorV2(always_apply=True),  # (H, W, C) --> (C, H, W)
    ])
    return A.Compose(transforms)


@torch.inference_mode()
def inference(model, class_model, image_paths, img_size, batch_size=24, device="cpu"):
    # retrieve number of images, computer number of batches
    num_images = len(image_paths)
    num_batches = (num_images + batch_size - 1) // batch_size
    
    # Define mean and std
    mean_seg = DatasetConfig.MEAN
    std_seg = DatasetConfig.STD
    mean_clf = DatasetConfig.MEAN_CLF
    std_clf = DatasetConfig.STD_CLF
    
    # Create a composition of preprocessing transformations for the segmentation model
    transforms  = setup_transforms(mean=mean_seg, std=std_seg)
    
    # list to collect overlaid paths
    overlay_paths = []

    # iterate over each batch 
    for batch_idx in range(num_batches):
        # capturing starting and ending index for each batch
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, num_images)
        batch_diff = end_idx - start_idx + 1
        # initiating lists for collecting batches of images
        batch_images_org = []         
        batch_images_clf = []
        batch_images_norm = []

        # iterate over each image in batch
        for idx in range(start_idx, end_idx):
            
            # Load and preprocess image for classification
            image_clf = load_file_linear(DatasetConfig.IMAGE_SIZE, image_paths[idx], depth=cv2.IMREAD_COLOR)
            image_clf = normalize_classif(image_clf, mean_clf, std_clf)
            
            # Load and preprocess image file for segmentation
            image_org = load_file_nearest(DatasetConfig.IMAGE_SIZE, image_paths[idx], depth=cv2.IMREAD_COLOR)
            # image_norm = transforms.ToTensor()(image_org) #-- that is from older version
            image_norm = transforms(image=image_org)["image"]
        
            # batch_image_org.append(image_org)
            batch_images_org.append(image_org)            
            batch_images_clf.append(image_clf)
            batch_images_norm.append(image_norm)
        
        # Classification predictions
        batch_images_clf = np.stack(batch_images_clf)
        y_pred_clf = class_model.predict(batch_images_clf, verbose=0).reshape(-1)
        clf_labels = y_pred_clf > DatasetConfig.THR
        true_idxs = np.where(clf_labels == True)[0]
        
        # Segmentation predictions
        batch_images_norm = torch.stack(batch_images_norm).to(device)
        mask_dim = (batch_diff, DatasetConfig.IMAGE_SIZE[0], DatasetConfig.IMAGE_SIZE[1])
        pred_all = torch.Tensor(np.zeros(mask_dim)).long()
        if len(true_idxs) > 0:
            predictions = model(batch_images_norm[true_idxs])
            pred_all[true_idxs] = torch.from_numpy(predictions.argmax(dim=1).cpu().numpy())
        
        # Apply overlay to each batch image
        for i in range(len(batch_images_org)):
            batch_img_np = np.float32(batch_images_org[i]) / 255.0
            pred_mask_rgb = num_to_rgb(pred_all[i], color_map=id2color)
            overlay_img = image_overlay(batch_img_np, pred_mask_rgb)
            
            # Get the filename from the original image path
            save_path = 'static/uploads/overlaid'
            mask_save_path = 'static/uploads/masks'  # New path for saving masks
            filename = os.path.splitext(os.path.basename(image_paths[start_idx + i]))[0]
            overlay_filename = os.path.join(save_path, f"{filename}_overlaid.png")
            mask_filename = os.path.join(mask_save_path, f"{filename}_mask.png")  # Mask filename
            
            # Save the overlaid image
            cv2.imwrite(overlay_filename, (overlay_img * 255).astype(np.uint8), [cv2.IMWRITE_PNG_COMPRESSION, 9])
            # Save the mask image
            cv2.imwrite(mask_filename, (pred_mask_rgb * 255).astype(np.uint8), [cv2.IMWRITE_PNG_COMPRESSION, 9])
            
            overlay_paths.append(overlay_filename)
            
    return overlay_paths


# retrieving segmentation model
def get_model(*, model_name, num_classes):
    model = SegformerForSemanticSegmentation.from_pretrained(
        model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )
    return model


# retrieving classification model
def get_class_model(class_model_loc, class_model):
    # classification model instantiation
    class_model_path = os.path.join(class_model_loc, class_model)
    class_model = load_model(class_model_path)
    return class_model
    
    
# loading checkpoint and model 
def predict(image_paths):
    
    CKPT_PATH = 'static/checkpoint.ckpt'
    model = MedicalSegmentationModel.load_from_checkpoint(CKPT_PATH)
            
    DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    model.to(DEVICE)
    model.eval()
    
    # predictions classification
    class_model = 'classification_model.keras'
    class_model_loc = 'static/'
    class_model = get_class_model(class_model_loc, class_model)

    predictions = inference(model, class_model, image_paths, img_size=DatasetConfig.IMAGE_SIZE, batch_size=10, device=DEVICE)
    
    return predictions

