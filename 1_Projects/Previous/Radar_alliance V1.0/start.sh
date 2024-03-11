#!/bin/bash


gnome-terminal -t "radar_remap_cpp" -x bash -c "cd ./radar_remap_cpp;
source install/setup.bash;
ros2 run radar_remap_cpp points_subscriber;" &

sleep 1

gnome-terminal -t "radar_yolov5_py" -x bash -c "cd ./radar_yolov5_py;
source install/setup.bash;
ros2 run radar_yolov5_py points_publisher;"
