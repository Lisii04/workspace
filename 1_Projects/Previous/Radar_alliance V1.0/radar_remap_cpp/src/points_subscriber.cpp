// [导入模块]
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

using namespace std;
using namespace cv;

// [全局变量]

// [读取小地图和ROI图像]>
string minimap_image_path = "./Images/minimap.png";
Mat minimap_image = imread(minimap_image_path);
string roi_image_path = "./Images/roi.jpg";
Mat roi_image = imread(roi_image_path);
Mat temp_image = imread(minimap_image_path);
// [读取小地图和ROI图像]<

// [定义变换矩阵]>
Mat_<float> high_transform_martix;
Mat_<float> low_transform_martix;
// [定义变换矩阵]<

// [定义坐标点]>
Mat_<float> image_point(3, 1);
Mat_<float> world_point_high(3, 1);
Mat_<float> world_point_low(3, 1);
// [定义坐标点]<

// [定义装甲板名称]>
std::vector<string> armor_names = { "1", "2", "3", "4", "5",
    "W", "1", "2", "3", "4", "5", "W" };
// [定义装甲板名称]<

// [定义存储map (机器人标签,机器人坐标)]>
std::map<int, cv::Point2f> cars_position;
// [定义存储map (机器人标签,机器人坐标)]<

// [透视变换函数]>
void remap(std::map<int, cv::Point2f> cars_position)
{

    minimap_image.copyTo(temp_image);
    try {

        for (size_t i = 0; i < cars_position.size(); i++) {
            // [由像素坐标计算到世界坐标]>>>
            // [读取机器人相机坐标]
            image_point = (Mat_<float>(3, 1) << cars_position[i].x, cars_position[i].y, 1);

            // [小地图显示未识别到的机器人]>
            if (cars_position[i].x == 0 && cars_position[i].y == 0) {
                if (i > 5) {
                    circle(temp_image, Point(50, 50 + 40 * i), 20, Scalar(255, 0, 0), -1);
                    string text = armor_names[i];
                    text.append(" [No Position Data]");
                    putText(temp_image, text, Point(34, 50 + 40 * i + 14), 1, 3, Scalar(255, 255, 255), 4);
                } else {
                    circle(temp_image, Point(50, 50 + 40 * i), 20, Scalar(0, 0, 255), -1);
                    string text = armor_names[i];
                    text.append(" [No Position Data]");
                    putText(temp_image, text, Point(34, 50 + 40 * i + 14), 1, 3, Scalar(255, 255, 255), 4);
                }
            }
            // [小地图显示未识别到的机器人]<

            // [矩阵计算]
            world_point_high = high_transform_martix * image_point;
            world_point_low = low_transform_martix * image_point;

            Point _world_point_high = Point(world_point_high.at<float>(0, 0) / world_point_high.at<float>(0, 2), world_point_high.at<float>(1, 0) / world_point_high.at<float>(0, 2));
            Point _world_point_low = Point(world_point_low.at<float>(0, 0) / world_point_low.at<float>(0, 2), world_point_low.at<float>(1, 0) / world_point_low.at<float>(0, 2));
            // [由像素坐标计算到世界坐标]<<<

            // [筛选负值]
            if (_world_point_high.x > 0 && _world_point_high.y > 0 && _world_point_low.x > 0 && _world_point_low.y > 0) {
                // [判断是否在高地 并绘制坐标]>
                if ((int)(roi_image.at<Vec3b>(_world_point_high.y, _world_point_high.x)[0]) > 150) {
                    if (i > 5) {
                        circle(temp_image, Point(_world_point_high.x, _world_point_high.y), 20, Scalar(255, 0, 0), -1);
                        putText(temp_image, armor_names[i], Point(_world_point_high.x - 16, _world_point_high.y + 14), 1, 3, Scalar(255, 255, 255), 4);

                    } else {
                        circle(temp_image, Point(_world_point_high.x, _world_point_high.y), 20, Scalar(0, 0, 255), -1);
                        putText(temp_image, armor_names[i], Point(_world_point_high.x - 16, _world_point_high.y + 14), 1, 3, Scalar(255, 255, 255), 4);
                    }

                } else {
                    if (i > 5) {
                        circle(temp_image, Point(_world_point_low.x, _world_point_low.y), 20, Scalar(255, 0, 0), -1);
                        putText(temp_image, armor_names[i], Point(_world_point_low.x - 16, _world_point_low.y + 14), 1, 3, Scalar(255, 255, 255), 4);

                    } else {
                        circle(temp_image, Point(_world_point_low.x, _world_point_low.y), 20, Scalar(0, 0, 255), -1);
                        putText(temp_image, armor_names[i], Point(_world_point_low.x - 16, _world_point_low.y + 14), 1, 3, Scalar(255, 255, 255), 4);
                    }
                }
                // [判断是否在高地 并绘制坐标]<
            }
        }
        imshow("1", temp_image);
        waitKey(1);
    } catch (const std::exception& e) {
        cout << "\033[31m[ERROR]\033[0m";
        std::cerr << e.what() << '\n';
    }
}
// [透视变换函数]<

// [ROS2 数据收发类]>
class Points_subscriber : public rclcpp::Node {
public:
    Points_subscriber(std::string name)
        : Node(name)
    {
        cout << ">\033[32m[DONE]\033[0m[节点启动]\n>\033[34m[WAITING]\033[0m\033[5m[等待数据]\033[0m" << endl;

        // [创建订阅]
        command_subscribe_ = this->create_subscription<std_msgs::msg::Float32MultiArray>("points_data", 10, std::bind(&Points_subscriber::command_callback, this, std::placeholders::_1));
    }

private:
    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr command_subscribe_;
    // [收到话题数据的回调函数]>
    void command_callback(const std_msgs::msg::Float32MultiArray::SharedPtr msg)
    {
        try {
            cout << "\033c>\033[33m[WORKING]\033[0m[正在接收点坐标]\033[?25l" << endl;
            if (msg->data.size() == 0) {
                cout << "\033[31m[ERROR]\033[0m[未识别到坐标]" << endl;
            } else {
                // [装填数据至 map<int, cv::Point2f> cars_position]>
                for (int i = 0; i < msg->data.size() - 2; i += 3) {
                    if (cars_position[int(msg->data.data()[i])].x == 0.0 && cars_position[int(msg->data.data()[i])].y == 0.0) {
                        cars_position[int(msg->data.data()[i])] = cv::Point2f(msg->data.data()[i + 1], msg->data.data()[i + 2]);
                    } else {
                        cars_position[int(msg->data.data()[i])] = cv::Point2f((cars_position[int(msg->data.data()[i])].x + msg->data.data()[i + 1]) / 2.0, (cars_position[int(msg->data.data()[i])].y + msg->data.data()[i + 2]) / 2.0);
                    }
                }
                // [装填数据至 map<int, cv::Point2f> cars_position]<

                // [输出信息]>
                for (int i = 0; i < cars_position.size(); i++) {
                    if (i == 0) {
                        cout << "\033[31m[RED]\n";
                    } else if (i == 6) {
                        cout << "\033[0m\033[34m[BLUE]\n";
                    }
                    cout << armor_names[i];
                    if (cars_position[i].x == 0.0 && cars_position[i].y == 0.0) {
                        printf(":[未识别到]\n");
                    } else {
                        printf(":(%0.1f,%0.1f)\n", cars_position[i].x, cars_position[i].y);
                    }
                }
                cout << "\033[0m\n"
                     << endl;
                // [输出信息]<

                // [透视变换]
                remap(cars_position);
            }
        } catch (const std::exception& e) {
            cout << "\033[31m[ERROR]\033[0m";
            std::cerr << e.what() << '\n';
        }
    }
    // [收到话题数据的回调函数]<
};
// [ROS2 数据收发类]<

int main(int argc, char** argv)
{
    cout << "\033[1m[正在启动小地图映射节点]\033[0m" << endl;
    cout << ">\033[33m[WORKING]\033[0m[初始化|读取地图和高地数据]" << endl;
    cout << ">\033[32m[DONE]\033[0m[读取结束]" << endl;

    // [读取相机标定矩阵]>
    try {
        string filename = "./Datas/martixs.yaml";
        cout << ">\033[33m[WORKING]\033[0m[读取相机标定矩阵]" << endl;

        // [以读取的模式打开相机标定文件]
        FileStorage fread(filename, FileStorage::READ);
        // [判断是否打开成功]
        if (!fread.isOpened()) {
            cout << ">\033[31m[ERROR]\033[0m[打开文件失败，请确认文件名称是否正确]" << endl;
            return -1;
        }

        // [读取Mat类型数据]
        fread["high_transform_martix"] >> high_transform_martix;
        fread["low_transform_martix"] >> low_transform_martix;

        fread.release();

        cout << ">\033[32m[DONE]\033[0m[读取结束]" << endl;
    } catch (const std::exception& e) {
        cout << "\033[31m[ERROR]\033[0m";
        std::cerr << e.what() << '\n';
    }

    // [读取相机标定矩阵]<

    // [调整ROI图像大小]
    resize(roi_image, roi_image, minimap_image.size());

    // [显示小地图窗口]
    namedWindow("1", 0);
    resizeWindow("1", Size(800, 500));

    // [初始化节点]
    cout << ">\033[33m[WORKING]\033[0m[启动节点]" << endl;
    rclcpp::init(argc, argv);

    // [创建对应节点的共享指针对象]
    auto node = std::make_shared<Points_subscriber>("points_subscriber");

    // [运行节点，并检测退出信号]
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
