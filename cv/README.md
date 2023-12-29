# Mimic - Computer Vision

## Overview

### Hand tracking
Uses MediaPipe's Hand Landmark Detection.

### Object detection
Uses custom forks of Facebook's Detic and detectron2 libraries.

There are two options for detecting objects:
- using built-in object labels (choose from 'lvis', 'objects365', 'openimages', or 'coco' datasets)
- using custom object labels (you can input your own classes to detect)

## Installation
Follow the root repository's installation instructions.

## Usage
### Demos
#### Hand tracking
Run `python hand_tracking.py` from the `cv/` directory.

#### Object detection
Navigate to the root repo, then run `python -m cv.demo.detic_demo`. Currently doesn't support other paths due to limitations in the Detic code.

To switch the target image, simply use an existing image from (or download a new one into)  `imgs/`. Then change the `image_path` variable to the name of the new image.

### Preprocessing

To execute the entire preprocessing code, navigate to the root repo and then run `python -m cv.preprocess`. Also doesn't support other paths at the moment.