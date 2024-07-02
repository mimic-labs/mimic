# Mimic
Framework for robots to learn simple human tasks with 1 video.

## Description

Our solution comprises a 3-step pipeline for enabling robots to mimic human tasks through video:
1. Feed robot with videos of human doing a basic task
2. Provide input through UI to identify important visual features
3. Robot mimics task, accounting for different configurations (e.g. cleaning, packing, cutting, etc.)

Currently, the code is broken down into a few main components:
* `cv/`: all visual preprocessing logic (hand tracking & object detection)
* `third_party/`: external packages, including Git submodules
* `simulation/`: MuJoCo testbe

## Getting Started

### Dependencies

* Python 3
* MuJoCo

### Installing

1. Run `git clone --recurse-submodules <repo-url>` and navigate to the cloned folder
2. (Optional) Create a virtual environment: `python -m venv venv` and activate it: `venv\Scripts\activate`
3. `pip install -r requirements.txt`

### Usage
See each subfolder for a more detailed README on how to execute the code in that module. yes

### Tracker Information
1. Make sure to have all detic submodules updated. Change the path to classifier in detic here: Detic/predict.py and Detic/detic/modeling/utils.py like this mimic/third_party/Detic/datasets/metadata/lvis_v1_clip_a+cname.npy

2. Download the Detic model and place in DETIC_ROOT/models/

Change path to Detic weights and yaml file in tracker.py
- Used the last model in Cross Dataset Evaluation here: https://github.com/facebookresearch/Detic/blob/main/docs/MODEL_ZOO.md 
- Get .yaml file from there too
- Pretty sure it is LVIS trained
- maybe we should check out box supervised (idk what that means)

3. run bash script/install.sh and bash script/download_ckpt.sh. 

- Change paths to weights files in Segment-and-Track-Anything/model_args.py; change path to grounding dino in detector.py

Download the two models and put int Segment-and-Track-Anything/ckpt/
- Get the SwinB-DeAOTL model from here https://github.com/yoxu515/aot-benchmark/blob/main/MODEL_ZOO.md

- download the SAM VIT huge and SAM VIT Large and SAM VIT base if you want to test different things


## TODOs:
- [x] clean code above and push to something usable by this bum arsh 
- [x] Run detic + sam on query image - detic_sam_init() in tracker
- [ ] use dino matching to match object mask i got in step 1 with step 2 masks - arsh you got it chief


<!-- ## Help

Any advise for common problems or issues.
```
command to run if program contains helper info
``` -->

## Authors

[Jack Hao](https://www.linkedin.com/in/jackhhao/)
<br>
[Arsh Singhal](https://www.linkedin.com/in/arsh-singhal/)

<!-- ## Version History

* 0.2
    * Various bug fixes and optimizations
    * See [commit change]() or See [release history]()
* 0.1
    * Initial Release

## License

This project is licensed under the [NAME HERE] License - see the LICENSE.md file for details

## Acknowledgments
