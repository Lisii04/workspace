#include <opencv4/opencv2/opencv.hpp>
#include <string>
#include <iostream>

using namespace cv;
using namespace std;

int main()
{
    VideoCapture inputVideo("../1.mp4");
    if (!inputVideo.isOpened())
    {
        cout << "Could not open the input video " << endl;
        return -1;
    }
    inputVideo.set(cv::CAP_PROP_FPS, 60);
    namedWindow("Camera", 0);
    resizeWindow("Camera", cv::Size(800, 600));

    Mat frame;
    string imgname;
    int f = 1;
    while (1)
    {
        inputVideo >> frame;
        if (frame.empty())
            break;
        imshow("Camera", frame);
        char key = waitKey(0);
        if (key == 'q' || key == 'Q') // 退出运行
            break;
        if (key == 'k' || key == 'K') // 截取图片
        {
            cout << "frame:" << f << endl;
            imgname = to_string(f++) + ".jpg";
            imwrite(imgname, frame);
        }
    }
    cout << "Finished writing" << endl;
    return 0;
}
