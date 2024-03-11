#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

using namespace cv;
using namespace std;

int main()
{
    VideoCapture inputVideo("../1.mp4");
    if (!inputVideo.isOpened()) {
        cout << "Could not open the input video " << endl;
        return -1;
    }
    // inputVideo.set(cv::CAP_PROP_FPS, 90);
    namedWindow("Camera", 1);

    Mat frame;
    string imgname;
    int count = 1;
    int savename = 1;
    while (1) {
        inputVideo >> frame;
        if (frame.empty())
            break;
        cout << "frame:" << count << endl;
        count++;
        if (count % 200 == 0 && count >= 0) {
            cout << "saved:" << count << endl;
            imgname = to_string(savename++) + ".jpg";
            imwrite(imgname, frame);
        }
    }
    cout << "Finished writing" << endl;
    return 0;
}
