![Project Image](images/README.webp)
# From 2D Medical Image Segmentation to 3D Model: From AI to VR

## About Us - DAST DATA team
We are four data science students from Spiced Academy, each of us coming from diverse academic backgrounds. This repository showcases our final project, which has been a collaborative effort among Tina Mangum, [Ali Rahjouei](https://github.com/arahjou), [Sergio Sanz](https://github.com/sergio-sanz-rodriguez), [Di Walsh](https://github.com/diwalsh). Our aim is to blend innovative technology and teamwork to tackle challenging problems in medical imaging.

## Introduction
In this project, we aim to develop a comprehensive pipeline that uses state-of-the-art machine learning methods, including Convolutional Neural Networks (CNNs) and Transformers, combined with 3D modeling techniques. Our system segments and visualizes CT scan images and is packaged into a user-friendly web-based application. Additionally, we integrate the segmentation results into a virtual reality environment, enhancing the interactive experience for medical professionals. This solution is designed not only to aid in automatic segmentation of medical imagery but also to serve as an innovative educational tool in medical training.

## Workflow
Our approach involved several key steps, starting from data sourcing to deploying a functional VR model:

- **Step 1:** Data Preparation - sourcing and preparing data from the "UW-Madison GI Tract Image Segmentation" Kaggle competition dataset, which includes annotations for three key organs: the stomach, small bowel, and large bowel.
- **Step 2:** Image Classification - training a CNN to classify images based on the presence or absence of the target organs.
- **Step 3:** Image Filtering - selecting relevant images for further processing.
- **Step 4:** Segmentation Model Training - utilizing a Transformer model with transfer learning techniques to segment the CT scans accurately.
- **Step 5:** Web Application Development - assembling the front-end and back-end components for seamless user interaction.
- **Step 6:** 3D Modeling and VR Integration - converting segmented images into 3D models and embedding them in a VR environment to enable immersive visualization.

## Models
- **Model A (Classification):** This model categorizes MRI images into two groups, identifying whether they contain any of the organs of interest.
- **Model B (Segmentation):** Following classification, this model segments the images, pinpointing the exact locations of the organs within the scans.

## Web Application
Our web application is built using Flask and integrates various technologies including JavaScript and Python. It acts as the interface where users can upload CT scans and view both the 2D segmentation results and the 3D visualizations.

## 3D Model and VR Integration
The 3D models are created using advanced rendering techniques and are compatible with Meta’s VR platforms. This allows for an interactive exploration of medical imagery in a virtual reality setting, providing a unique educational and diagnostic tool that can be accessed worldwide.

## Used Deep Learning Platforms

## Notebooks

#### [3D_model](notebooks/3D_model.ipynb)
#### [](model_evaluation.ipynb)
#### [](prepare_data_classification.ipynb)
##### [](prepare_data_classification.py)
#### [](added script to prepare data for the binary classification model
#### [](prepare_data_segmentation.ipynb
#### [](prepare_data_segmentation.py
#### [](Segformer_medical_lightning.ipynb

## References

[UW-Madison GI Tract Image Segmentation](https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation/)  

[Learn OpenCV](https://learnopencv.com/medical-image-segmentation/)  
