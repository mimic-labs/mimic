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
* `simulation/`: MuJoCo testbed

## Getting Started

### Dependencies

* Python 3
* MuJoCo

### Installing

1. Run `git clone --recurse-submodules <repo-url>` and navigate to the cloned folder
2. (Optional) Create a virtual environment: `python -m venv venv` and activate it: `venv\Scripts\activate`
3. `pip install -r requirements.txt`

### Usage
See each subfolder for a more detailed README on how to execute the code in that module.

<!-- ## Help

Any advise for common problems or issues.
```
command to run if program contains helper info
``` -->

## Authors

[Jack Hao](https://www.linkedin.com/in/jackhhao/)
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