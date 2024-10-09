# SAM Segmentator

SAM Segmentator is an GUI application created for educational segmetational purposes.
Segmentator outputs: 
  - Original image
  - Mask
  - Txt annotation (for models like YOLO)

## Table of Contents
[Installation](#installation)
[Usage](#usage)
[Features](#features)

## Installation

00. Setup CUDA and CUDNN
   ```bash
   https://developer.nvidia.com/cuda-toolkit
   https://developer.nvidia.com/cudnn

1. Clone the repo:
   ```bash
   git clone https://github.com/lukasiktar/SAM_segmentator.git

2. Download the SAM model from:
   ```bash
   https://github.com/facebookresearch/segment-anything.git

3. Install Opencv library from:
   ```bash
    https://github.com/opencv/opencv.git

4. Build torch and torchvision from (it is reccomended to build from source with CUDA support for your CUDA version):
   ```bash
   https://pytorch.org/get-started/previous-versions/

5. Install Tkinter
   ```bash
   sudo apt install python3-tk

6. Install segment-anything and albumentations
   ```bash
   pip install segment-anything albumentations

7. Download the repository and store it in the working directory:
   ```bash
   git clone https://github.com/OpenGVLab/SAM-Med2D.git
   

## Usage

To start the application, build the Python executable and start it.

1. Choose the apropriate image
2. Draw the bounding box around specified object or click on it
3. Perform segemetation using SAM
4. Edit segmentation (if neccessary)
5. Save the segmetation results

## Features

Segment using Point prompt
Segment using Box prompt
Edit the segmentation
Accept or Reject segmentation

