# ECE-276A-Project1
## Orientation Tracking
This is the project 1 of the course UCSD ECE276A: Sensing & Estimation in Robotics.

1. Orientation Tracking: Implement a projected gradient descent algorithm to track the 3-D orientation of a rotating body using readings from an inertial measurement unit (IMU). 
2. Panorama: Using your orientation estimates, generate a panoramic image by stitching camera images obtained by the rotating body.

## Usage:
### Install package:
    pip3 install -r requirement.txt
### Run code:
    python3 main.py [problem] [dataset] [mode]
### Example:
    python3 main.py 1 1 train
    python3 main.py 2 2 test

*NOTE: mode == train would create image with both imu and vicon, while using mode == test would create image imu only*

### Source code description:
- **code/main.py**: Main function.
- **code/quaternion.py**: Functions for quaternion operations and IMU calibration.
- **code/panorama.py**: Functions for generating the panorama image.
- **code/utils.py**: Functions for loading the data and others.
- **code/rotplot.py**: Functions for visualizeing the orientation.
    
