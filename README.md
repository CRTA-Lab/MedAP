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

1. Clone the repo:
   ```bash
   git clone https://github.com/lukasiktar/SAM_segmentator.git

2. Download the SAM model from:
   ```bash
   https://github.com/facebookresearch/segment-anything.git

3. Install Opencv library from:
   ```bash
    https://github.com/opencv/opencv.git

4. Build torch and torchvision from
   ```bash
   https://pytorch.org/get-started/previous-versions/

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

