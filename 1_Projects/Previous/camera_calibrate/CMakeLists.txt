# cmake needs this line
cmake_minimum_required(VERSION 3.1)

# Define project name
project(Opencv-camera-pnpsolve)

# Find OpenCV, you may need to set OpenCV_DIR variable
# to the absolute path to the directory containing OpenCVConfig.cmake file
# via the command line or GUI
find_package(OpenCV REQUIRED)

# Enable C++11
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# Declare the executable target built from your sources
add_executable(Opencv-camera-pnpsolve camera_calibration.cpp)
# add_executable(Opencv-camera-pnpsolve chesscoard_capture.cpp)

# Link your application with OpenCV libraries
target_link_libraries(Opencv-camera-pnpsolve PRIVATE ${OpenCV_LIBS})
