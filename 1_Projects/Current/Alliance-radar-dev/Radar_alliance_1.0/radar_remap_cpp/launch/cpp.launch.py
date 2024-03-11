from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    points_subscriber = Node(
        package="radar_remap_cpp",
        executable="points_subscriber"
    )
    launch_description = LaunchDescription(
        [points_subscriber])
    return launch_description
