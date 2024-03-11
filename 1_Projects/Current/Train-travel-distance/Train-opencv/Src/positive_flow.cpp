// lucas_kanade.cpp
#include <cmath>
#include <functional>
#include <iostream>
#include <math.h>
#include <opencv2/core.hpp>
#include <opencv2/core/operations.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/optflow.hpp>
#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>
#include <queue>
#include <string>
#include <sys/stat.h>
using namespace cv;
using namespace std;

cv::Mat ROI_extract(cv::Mat inputFrame, std::vector<cv::Point> points)
{
    cv::Mat dst, src;
    src = inputFrame;
    cv::Mat ROI = cv::Mat::zeros(src.size(), CV_8UC1);
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Point> pts;

    for (int i = 0; i < points.size(); i++) {
        pts.push_back(points[i]);
    }

    contours.push_back(pts);
    drawContours(ROI, contours, 0, cv::Scalar(255), -1);
    src.copyTo(dst, ROI);

    return dst;
}

double getDistance(cv::Point pointO, cv::Point pointA)
{
    double distance;
    distance = powf((pointO.x - pointA.x), 2) + powf((pointO.y - pointA.y), 2);
    distance = sqrtf(distance);

    return distance;
}

int lucas_kanade(const string& filename)
{
    VideoCapture capture(filename);
    if (!capture.isOpened()) {
        //打开视频输入错误
        cerr << "Unable to open file!" << endl;
        return 0;
    }
    // 创建一些随机的颜色
    vector<Scalar> colors;
    RNG rng;
    for (int i = 0; i < 100; i++) {
        int r = rng.uniform(0, 256);
        int g = rng.uniform(0, 256);
        int b = rng.uniform(0, 256);
        colors.push_back(Scalar(r, g, b));
    }
    Mat old_frame, old_gray;
    vector<Point2f> p0, p1;

    queue<double> speeds;
    // 取第一帧并在其中找到角点
    capture >> old_frame;

    std::vector<cv::Point> points;
    double max_X = old_frame.size().width;
    double max_Y = old_frame.size().height;
    points.push_back(cv::Point(max_X * (2.0 / 5.0), max_Y * (1.2 / 5.0))); // LD
    points.push_back(cv::Point(max_X * (2.5 / 5.0), max_Y * (1.2 / 5.0))); // RD
    points.push_back(cv::Point(max_X * (2.6 / 5.0), max_Y * (0.0 / 5.0))); // RU
    points.push_back(cv::Point(max_X * (1.5 / 5.0), max_Y * (0.0 / 5.0))); // LU
    // old_frame = ROI_extract(old_frame, points);

    cv::Mat remap = cv::Mat::zeros(cv::Size(200, 500), CV_8UC1);
    cv::Point2f AffinePoints[4] = { points[0], points[1], points[2], points[3] }; //变化前的4个节点
    cv::Point2f transformed_points[4] = { cv::Point(0, remap.rows), cv::Point(remap.cols, remap.rows), cv::Point(remap.cols, 0), cv::Point(0, 0) }; //变化后的4个节点
    cv::Mat Trans = cv::getPerspectiveTransform(AffinePoints, transformed_points);
    cv::warpPerspective(old_frame, remap, Trans, cv::Size(remap.cols, remap.rows));
    old_frame = remap;

    cvtColor(old_frame, old_gray, COLOR_BGR2GRAY);
    goodFeaturesToTrack(old_gray, p0, 100, 0.25, 7, Mat(), 7, false, 0.04);
    // 创建用于绘图的掩模图像
    Mat mask = Mat::zeros(old_frame.size(), old_frame.type());
    int counter = 0;
    while (true) {
        Mat frame, frame_gray;
        capture >> frame;
        if (frame.empty())
            break;
        cout << "Processing: " << counter << "\n";
        if (counter < 2350) {
            counter++;
            continue;
        }
        imshow("ori", frame);
        cv::warpPerspective(frame, remap, Trans, cv::Size(remap.cols, remap.rows));
        frame = remap;
        // frame = ROI_extract(frame, points);

        cvtColor(frame, frame_gray, COLOR_BGR2GRAY);

        if (counter % 10 == 0 || p0.size() == 0) {
            mask = Mat::zeros(old_frame.size(), old_frame.type());
            p0.clear();
            p1.clear();
            goodFeaturesToTrack(old_gray, p0, 100, 0.25, 7, Mat(), 7, false, 0.04);
        }

        // 计算光流
        vector<uchar> status;
        vector<float> err;
        TermCriteria criteria = TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 10, 0.03);
        calcOpticalFlowPyrLK(old_gray, frame_gray, p0, p1, status, err, Size(15, 15), 2, criteria);
        vector<Point2f> good_new;

        double distance = 0;
        double count = 0;

        for (uint i = 0; i < p0.size(); i++) {

            // 选择比较好的点
            if (status[i] == 1) {
                good_new.push_back(p1[i]);
                if (p1[i].y >= 200) {
                    // 画出轨迹
                    line(mask, p1[i], p0[i], colors[i], 2);
                    circle(frame, p1[i], 5, colors[i], -1);

                    distance += getDistance(p0[i], p1[i]);
                    count += 1;
                }
            }
        }
        double speed = distance / count;
        if (speeds.size() < 30) {
            speeds.push(speed);
        } else {
            speeds.push(speed);
            speeds.pop();
        }
        speed = 0;
        for (int i = 0; i < speeds.size(); i++) {
            speed += speeds.front();
            speeds.push(speeds.front());
            speeds.pop();
        }
        speed = speed / (double)speeds.size();

        putText(frame, "Speed:", Point(10, 20), 1, 1.6, Scalar(255, 255, 0), 2);
        string text;
        text.append(to_string(speed));
        text.append("p/s");
        putText(frame, text, Point(10, 50), 1, 1.6, Scalar(255, 255, 0), 2);

        Mat img;
        add(frame, mask, img);

        imshow("flow", img);
        int keyboard = waitKey(0);
        if (keyboard == 'q' || keyboard == 27)
            break;
        // 创建用于绘图的掩模图像
        old_gray = frame_gray.clone();
        p0 = good_new;

        counter++;
    }
    return 1;
}

int main(int argc, char** argv)
{
    string filename = "../video.mp4";

    string method = "lucaskanade";

    bool to_gray = true;
    lucas_kanade(filename);
    return 0;
}
