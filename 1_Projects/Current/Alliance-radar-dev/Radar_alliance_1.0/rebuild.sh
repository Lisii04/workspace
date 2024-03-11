#!/bin/bash

source /opt/ros/humble/setup.bash
cd radar_remap_cpp
rm -r ./build
colcon build
source install/setup.bash
cd ../radar_yolov5_py
rm -r ./build
colcon build
source install/setup.bash
cd ..
