// [导入模块]
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <fstream>
#include <time.h>

using namespace std;
using namespace cv;

// [全局变量]
#define MODE 1

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
std::vector<string> armor_names = {"1", "2", "3", "4", "5",
                                   "W", "1", "2", "3", "4", "5", "W"};
// [定义装甲板名称]<

// [定义存储map (机器人标签,机器人坐标)]>
std::map<int, cv::Point2f> cars_position;
// [定义存储map (机器人标签,机器人坐标)]<

// [日志实现]>
class Logger
{
public:
    enum log_level
    {
        debug,
        info,
        warning,
        error
    }; // 日志等级
    enum log_target
    {
        file,
        terminal,
        file_and_terminal
    }; // 日志输出目标
public:
    Logger();
    Logger(log_target target, log_level level, const std::string &path);
    ~Logger();

    void DEBUG(const std::string &text);
    void INFO(const std::string &text);
    void WARNING(const std::string &text);
    void ERRORS(const std::string &text);

private:
    std::ofstream m_outfile;                                   // 将日志输出到文件的流对象
    log_target m_target;                                       // 日志输出目标
    std::string m_path;                                        // 日志文件路径
    log_level m_level;                                         // 日志等级
    void output(const std::string &text, log_level act_level); // 输出行为
};
std::string currTime()
{
    // 获取当前时间，并规范表示
    char tmp[64];
    time_t ptime;
    time(&ptime); // time_t time (time_t* timer);
    strftime(tmp, sizeof(tmp), "%Y-%m-%d %H:%M:%S", localtime(&ptime));
    return tmp;
}
Logger::Logger()
{
    // 默认构造函数
    m_target = terminal;
    m_level = debug;
}
Logger::Logger(log_target target, log_level level, const std::string &path)
{
    m_target = target;
    m_path = path;
    m_level = level;

    std::string strContent = "[" + currTime() + "]" + "<开始日志>\n";
    if (target != terminal)
    {
        m_outfile.open(path, std::ios::out | std::ios::app); // 打开输出文件
        m_outfile << strContent;
    }
    if (target != file)
    {
        // 如果日志对象不是仅文件
        std::cout << strContent;
    }
}
Logger::~Logger()
{
    std::string strContent = "[" + currTime() + "]" + "<结束日志>\n";
    if (m_outfile.is_open())
    {
        m_outfile << strContent;
    }
    m_outfile.flush();
    m_outfile.close();
}
void Logger::DEBUG(const std::string &text)
{
    output(text, debug);
}
void Logger::INFO(const std::string &text)
{
    output(text, info);
}
void Logger::WARNING(const std::string &text)
{
    output(text, warning);
}
void Logger::ERRORS(const std::string &text)
{
    output(text, error);
}
void Logger::output(const std::string &text, log_level act_level)
{
    std::string prefix;
    if (act_level == debug)
        prefix = "[DEBUG] ";
    else if (act_level == info)
        prefix = "[INFO] ";
    else if (act_level == warning)
        prefix = "[WARNING] ";
    else if (act_level == error)
        prefix = "[ERROR] ";
    // else prefix = "";
    // prefix += __FILE__;
    // prefix += " ";
    std::string outputContent = prefix + currTime() + " : " + text + "\n";
    if (m_level <= act_level && m_target != file)
    {
        // 当前等级设定的等级才会显示在终端，且不能是只文件模式
        std::cout << outputContent;
    }
    if (m_target != terminal)
        m_outfile << outputContent;

    m_outfile.flush(); // 刷新缓冲区
}
// [日志实现]<
Logger logger(Logger::file, Logger::debug, "./logs/cpp.log");

// [透视变换函数]>
void remap(std::map<int, cv::Point2f> cars_position)
{

    minimap_image.copyTo(temp_image);
    try
    {

        for (size_t i = 0; i < cars_position.size(); i++)
        {
            // [由像素坐标计算到世界坐标]>>>
            // [读取机器人相机坐标]
            image_point = (Mat_<float>(3, 1) << cars_position[i].x, cars_position[i].y, 1);

            // [小地图显示未识别到的机器人]>
            if (cars_position[i].x == 0 && cars_position[i].y == 0)
            {
                if (i > 5)
                {
                    circle(temp_image, Point(50, 50 + 40 * i), 20, Scalar(255, 0, 0), -1);
                    string text = armor_names[i];
                    text.append(" [No Position Data]");
                    putText(temp_image, text, Point(34, 50 + 40 * i + 14), 1, 3, Scalar(255, 255, 255), 4);
                }
                else
                {
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
            if (_world_point_high.x > 0 && _world_point_high.y > 0 && _world_point_low.x > 0 && _world_point_low.y > 0)
            {
                // [判断是否在高地 并绘制坐标]>
                if ((int)(roi_image.at<Vec3b>(_world_point_high.y, _world_point_high.x)[0]) > 150)
                {
                    if (i > 5)
                    {
                        circle(temp_image, Point(_world_point_high.x, _world_point_high.y), 20, Scalar(255, 0, 0), -1);
                        putText(temp_image, armor_names[i], Point(_world_point_high.x - 16, _world_point_high.y + 14), 1, 3, Scalar(255, 255, 255), 4);
                    }
                    else
                    {
                        circle(temp_image, Point(_world_point_high.x, _world_point_high.y), 20, Scalar(0, 0, 255), -1);
                        putText(temp_image, armor_names[i], Point(_world_point_high.x - 16, _world_point_high.y + 14), 1, 3, Scalar(255, 255, 255), 4);
                    }
                }
                else
                {
                    if (i > 5)
                    {
                        circle(temp_image, Point(_world_point_low.x, _world_point_low.y), 20, Scalar(255, 0, 0), -1);
                        putText(temp_image, armor_names[i], Point(_world_point_low.x - 16, _world_point_low.y + 14), 1, 3, Scalar(255, 255, 255), 4);
                    }
                    else
                    {
                        circle(temp_image, Point(_world_point_low.x, _world_point_low.y), 20, Scalar(0, 0, 255), -1);
                        putText(temp_image, armor_names[i], Point(_world_point_low.x - 16, _world_point_low.y + 14), 1, 3, Scalar(255, 255, 255), 4);
                    }
                }
                // [判断是否在高地 并绘制坐标]<
            }
        }
        imshow("1", temp_image);
        waitKey(1);
    }
    catch (const std::exception &e)
    {
        string errors = "[ERROR]";
        errors.append(e.what());
        logger.ERRORS(errors);
    }
}
// [透视变换函数]<

// [显示函数]>
void display(std::map<int, cv::Point2f> cars_position)
{
    minimap_image.copyTo(temp_image);
    try
    {

        for (size_t i = 0; i < cars_position.size(); i++)
        {

            // [小地图显示未识别到的机器人]>
            if (cars_position[i].x == 0 && cars_position[i].y == 0)
            {
                if (i > 5)
                {
                    circle(temp_image, Point(50, 50 + 40 * i), 20, Scalar(255, 0, 0), -1);
                    string text = armor_names[i];
                    text.append(" [No Position Data]");
                    putText(temp_image, text, Point(34, 50 + 40 * i + 14), 1, 3, Scalar(255, 255, 255), 4);
                }
                else
                {
                    circle(temp_image, Point(50, 50 + 40 * i), 20, Scalar(0, 0, 255), -1);
                    string text = armor_names[i];
                    text.append(" [No Position Data]");
                    putText(temp_image, text, Point(34, 50 + 40 * i + 14), 1, 3, Scalar(255, 255, 255), 4);
                }
            }
            // [小地图显示未识别到的机器人]<

            circle(temp_image, Point(cars_position[i].x, cars_position[i].y), 20, Scalar(255, 0, 0), -1);
            putText(temp_image, armor_names[i], Point(cars_position[i].x - 16, cars_position[i].y + 14), 1, 3, Scalar(255, 255, 255), 4);
        }
        imshow("1", temp_image);
        waitKey(1);
    }
    catch (const std::exception &e)
    {
        string errors = "[ERROR]";
        errors.append(e.what());
        logger.ERRORS(errors);
    }
}
// [显示函数]<

// [ROS2 数据收发类]>
class Points_subscriber : public rclcpp::Node
{
public:
    Points_subscriber(std::string name)
        : Node(name)
    {
        string topic;
        if (MODE)
        {
            topic = "car_position";
        }
        else
        {
            topic = "points_data";
        }
        // [创建订阅]
        command_subscribe_ = this->create_subscription<std_msgs::msg::Float32MultiArray>(topic, 10, std::bind(&Points_subscriber::command_callback, this, std::placeholders::_1));
    }

private:
    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr command_subscribe_;
    // [收到话题数据的回调函数]>
    void command_callback(const std_msgs::msg::Float32MultiArray::SharedPtr msg)
    {
        try
        {
            logger.INFO("[正在接收点坐标]");
            if (msg->data.size() == 0)
            {
                logger.WARNING("[WARNING][未识别到坐标]");
            }
            else
            {
                if (MODE)
                {
                    // [装填数据至 map<int, cv::Point2f> cars_position]>
                    for (int i = 0; i < msg->data.size() - 2; i += 3)
                    {
                        cars_position[i / 3] = cv::Point2f(msg->data.data()[i + 1], msg->data.data()[i + 2]);
                    }
                    // [装填数据至 map<int, cv::Point2f> cars_position]<

                    // [显示数据]
                    display(cars_position);
                }
                else
                {
                    // [装填数据至 map<int, cv::Point2f> cars_position]>
                    for (int i = 0; i < msg->data.size() - 2; i += 3)
                    {
                        if (cars_position[int(msg->data.data()[i])].x == 0.0 && cars_position[int(msg->data.data()[i])].y == 0.0)
                        {
                            cars_position[int(msg->data.data()[i])] = cv::Point2f(msg->data.data()[i + 1], msg->data.data()[i + 2]);
                        }
                        else
                        {
                            cars_position[int(msg->data.data()[i])] = cv::Point2f((cars_position[int(msg->data.data()[i])].x + msg->data.data()[i + 1]) / 2.0, (cars_position[int(msg->data.data()[i])].y + msg->data.data()[i + 2]) / 2.0);
                        }
                    }
                    // [装填数据至 map<int, cv::Point2f> cars_position]<

                    // [透视变换]
                    remap(cars_position);
                }
            }
        }
        catch (const std::exception &e)
        {
            string errors = "[ERROR]";
            errors.append(e.what());
            logger.ERRORS(errors);
        }
    }
    // [收到话题数据的回调函数]<
};
// [ROS2 数据收发类]<

int main(int argc, char **argv)
{
    string mode = "[当前模式]";
    mode.append(std::to_string(MODE));
    logger.INFO(mode);
    logger.INFO("[正在启动小地图映射节点]");
    logger.INFO("[初始化|读取地图和高地数据]");
    logger.INFO("[读取结束]");

    if (MODE)
    {
        namedWindow("1", 0);
        resizeWindow("1", Size(1920, 1080));
    }
    else
    {
        // [读取相机标定矩阵]>
        try
        {
            string filename = "./Datas/martixs.yaml";
            logger.INFO("[读取相机标定矩阵]");

            // [以读取的模式打开相机标定文件]
            FileStorage fread(filename, FileStorage::READ);
            // [判断是否打开成功]
            if (!fread.isOpened())
            {
                logger.ERRORS("[ERROR][打开文件失败，请确认文件名称是否正确]");
                return -1;
            }

            // [读取Mat类型数据]
            fread["high_transform_martix"] >> high_transform_martix;
            fread["low_transform_martix"] >> low_transform_martix;

            fread.release();

            logger.INFO("[读取结束]");
        }
        catch (const std::exception &e)
        {
            string errors = "[ERROR]";
            errors.append(e.what());
            logger.ERRORS(errors);
        }
        // [读取相机标定矩阵]<
        // [调整ROI图像大小]
        resize(roi_image, roi_image, minimap_image.size());

        // [显示小地图窗口]
        namedWindow("1", 0);
        resizeWindow("1", Size(800, 500));
    }

    // [初始化节点]
    logger.INFO("[启动完成]");
    rclcpp::init(argc, argv);

    // [创建对应节点的共享指针对象]
    auto node = std::make_shared<Points_subscriber>("points_subscriber");

    // [运行节点，并检测退出信号]
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
