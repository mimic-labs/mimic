# Mimic - Third Party

## Overview

### [detectron2](https://github.com/facebookresearch/detectron2)
Base library used by Detic for detection & segmentation.

[Our fork](https://github.com/jackhhao/detectron2) also displays the bounding boxes for the detected object instances.

### [Detic](https://github.com/facebookresearch/Detic/)
Detector with image label support & custom zero-shot detection.

[Our fork](https://github.com/jackhhao/Detic) eliminates the need for manually inserting their third party library paths into `sys.path` at runtime. Instead, we create setup files for those libraries and install them directly into the Python environment.