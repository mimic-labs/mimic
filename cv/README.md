# Mimic - Computer Vision

## Overview

### Hand tracking
Uses MediaPipe's Hand Landmark Detection.

### Object detection
Uses custom forks of Facebook's Detic and detectron2 libraries.

## Installation
1. `git lfs install` (if Git LFS is not already installed)
2. `git clone --recurse-submodules <repo-url>`
3. `pip install -r requirements.txt`

## Usage
### Hand tracking
Run `hand_tracking.py`.

### Object detection
Run `detic_demo.py` from within the `cv` directory. Currently doesn't support other paths.

To switch the target image, simply use an existing image from (or download a new one into)  `imgs/`. Then change the `image_path` variable.

There are two options for detecting objects:
- using built-in object labels (choose from 'lvis', 'objects365', 'openimages', or 'coco' datasets)
- using custom object labels (you can input your own classes to detect)