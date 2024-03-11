from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    points_publisher = Node(
        package="radar_yolov5_py",
        executable="points_publisher"
    )
    launch_description = LaunchDescription(
        [points_publisher])
    return launch_description
