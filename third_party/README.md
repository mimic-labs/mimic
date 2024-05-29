# Mimic - Third Party

## Overview

### [detectron2](https://github.com/facebookresearch/detectron2)
Base library used by Detic for detection & segmentation.

[Our fork](https://github.com/jackhhao/detectron2) also displays the bounding boxes for the detected object instances.

### [Detic](https://github.com/facebookresearch/Detic/)
Detector with image label support & custom zero-shot detection.

[Our fork](https://github.com/jackhhao/Detic) eliminates the need for manually inserting their third party library paths into `sys.path` at runtime. Instead, we create setup files for those libraries and install them directly into the Python environment.

### [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM)
Detector with image label support & custom zero-shot detection.

[Our fork](https://github.com/mimic-labs/FastSAM) eliminates the need for manually inserting their third party library paths into `sys.path` at runtime. Instead, we create setup files for those libraries and install them directly into the Python environment.

## Installation

To use the third party libraries, simply install them as editable `pip` packages. Run the following commands from the root directory of the project:

```bash
cd third_party/<submodule name>
pip install -e .
```

To access any of a library's scripts that are located outside of the installed Python package, first navigate to the project's root directory, then import `third_party.<submodule name>` to access the library's files.

```python
import os
os.chdir('..')  # Navigate to the project's root directory
import third_party.<submodule name>.<script name> # third_party is located under root
```