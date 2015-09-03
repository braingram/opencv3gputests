#include <time.h>
#include <sys/time.h>
#include <iostream>
#include "opencv2/core/core.hpp"

using namespace std;
using namespace cv;

int main() {
    struct timeval start_time, end_time, delta_time;

    // make mat on cpu
    Mat a(2048, 2048, CV_32F, Scalar(0.));
    Mat b(2048, 2048, CV_32F, Scalar(0.));
    cout << "Made mat of " << a.size() << endl;
    randu(a, Scalar(0), Scalar(100));
    randu(b, Scalar(0), Scalar(100));
    cout << "filled with random values" << endl;

    double min_value;
    double max_value;

    minMaxLoc(a, &min_value, &max_value);
    cout << "a: min: " << min_value << " max: " << max_value << endl;
    minMaxLoc(b, &min_value, &max_value);
    cout << "b: min: " << min_value << " max: " << max_value << endl;

    cout << "multiply a * b " << endl;
    gettimeofday(&start_time, NULL);

    multiply(a, b, b);

    gettimeofday(&end_time, NULL);
    timersub(&end_time, &start_time, &delta_time);
    cout << "multiply took: " << (delta_time.tv_usec + 1000000 * delta_time.tv_sec) / 1000. << endl;

    minMaxLoc(b, &min_value, &max_value);
    cout << "b: min: " << min_value << " max: " << max_value << endl;
}
