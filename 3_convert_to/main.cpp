#include <time.h>
#include <sys/time.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#define N 10

cv::Mat im;
cv::Mat fim = cv::Mat(2048, 2048, CV_32F);

struct timeval start, end, delta;

int main() {
    im = cv::imread("../data/0.tif");

    for (int i=0; i < N; i++) {
        gettimeofday(&start, NULL);
        im.convertTo(fim, CV_32F);
        gettimeofday(&end, NULL);
        timersub(&end, &start, &delta);
        std::cout << (delta.tv_usec + 1000000 * delta.tv_sec) / 1000. << std::endl;
    }

    im.release();
    fim.release();
}
