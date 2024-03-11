#!/bin/bash

source /opt/ros/humble/setup.bash
# cd ../ROS-TCP-Endpoint
# bash launch.sh
# cd ../Radar_alliance_1.0
cd radar_remap_cpp
source install/setup.bash
cd ../radar_yolov5_py
source install/setup.bash
cd ../bringup/logs
rm -f cpp.log py.log
touch cpp.log py.log
cd ..
ros2 launch allpkg.launch.py
