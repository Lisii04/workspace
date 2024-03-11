#!/bin/bash

source /opt/ros/humble/setup.bash
cd radar_remap_cpp
colcon build
source install/setup.bash
cd ../radar_yolov5_py
colcon build
source install/setup.bash
cd ../bringup/logs
rm -f cpp.log py.log
touch cpp.log py.log
cd ..
ros2 launch allpkg.launch.py
