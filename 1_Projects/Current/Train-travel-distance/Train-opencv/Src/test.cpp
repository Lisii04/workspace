#include <iostream>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv4/opencv2/opencv.hpp>
#include <vector>

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

int main()
{
    int frame_count = 0;
    std::queue<cv::Point2f> all_points;

    cv::VideoCapture video;
    video.open("../video.mp4");
    while (true) {
        cv::Mat frame;
        video >> frame;
        frame_count++;
        std::cout << frame_count << "\n";

        if (frame.empty()) {
            break;
        } else if (frame_count < 1200) {
            continue;
        }

        //集中定义
        cv::Mat Hsv, Hsv_green, bulr, thres, canny, pr_frame, pr_gray;

        std::vector<cv::Point> points;
        double max_X = frame.size().width;
        double max_Y = frame.size().height;
        points.push_back(cv::Point(max_X * (2.0 / 5.0), max_Y * (1.2 / 5.0))); // LD
        points.push_back(cv::Point(max_X * (2.5 / 5.0), max_Y * (1.2 / 5.0))); // RD
        points.push_back(cv::Point(max_X * (2.6 / 5.0), max_Y * (0.0 / 5.0))); // RU
        points.push_back(cv::Point(max_X * (1.5 / 5.0), max_Y * (0.0 / 5.0))); // LU
        cv::Mat Roi = ROI_extract(frame, points);

        cv::Mat remap = cv::Mat::zeros(cv::Size(200, 400), CV_8UC1);
        cv::Point2f AffinePoints[4] = { points[0], points[1], points[2], points[3] }; //变化前的4个节点
        cv::Point2f transformed_points[4] = { cv::Point(0, remap.rows), cv::Point(remap.cols, remap.rows), cv::Point(remap.cols, 0), cv::Point(0, 0) }; //变化后的4个节点

        cv::Mat Trans = cv::getPerspectiveTransform(AffinePoints, transformed_points);
        cv::warpPerspective(frame, remap, Trans, cv::Size(remap.cols, remap.rows));

        cv::imshow("1", remap);
        cv::waitKey(0);
    }

    std::cout << "Done"
              << "\n";
    return 0;
}
