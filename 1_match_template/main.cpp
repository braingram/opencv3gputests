#include <time.h>
#include <sys/time.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudaimgproc.hpp"

#define N 10


//using namespace std;
//using namespace cv;


struct timeval start_time, end_time;
struct timeval delta_times[N];
double mean_delta_time;

double min_value, max_value;
cv::Point min_point, max_point;

cv::Mat a(2048, 2048, CV_32F, cv::Scalar(0.));
cv::Mat b(2048, 2048, CV_32F, cv::Scalar(0.));
cv::Mat c(2048, 2048, CV_32F, cv::Scalar(0.));
cv::Mat d(2048, 2048, CV_32F, cv::Scalar(0.));


double time_gpu() {
    mean_delta_time = 0;
    cv::randu(a, cv::Scalar(0), cv::Scalar(100));
    cv::randu(b, cv::Scalar(0), cv::Scalar(100));
    cv::randu(c, cv::Scalar(0), cv::Scalar(100));
    cv::randu(d, cv::Scalar(0), cv::Scalar(100));
    
    cv::cuda::GpuMat ga(2048, 2048, CV_32F);
    cv::cuda::GpuMat gb(2048, 2048, CV_32F);
    cv::cuda::GpuMat gc(2048, 2048, CV_32F);
    cv::cuda::GpuMat gd(2048, 2048, CV_32F);

    ga.upload(a);
    gb.upload(b);
    gc.upload(c);
    gd.upload(d);

    cv::Rect tcrop(824, 824, 400, 400);
    cv::Rect mcrop(774, 774, 500, 500);

    cv::cuda::GpuMat mt_result(500-400+1, 500-400+1, CV_32F);

    cv::Ptr<cv::cuda::TemplateMatching> tm = cv::cuda::createTemplateMatching(CV_32F, CV_TM_CCORR);

    // transfer both the gpu
    for (int i=0; i < N; i++) {
        gettimeofday(&start_time, NULL);

        // crop images
        cv::cuda::GpuMat ca(ga, tcrop);
        cv::cuda::GpuMat cb(gb, mcrop);
        cv::cuda::GpuMat cc(gc, mcrop);
        cv::cuda::GpuMat cd(gd, mcrop);

        tm->match(cb, ca, mt_result);
        cv::cuda::minMaxLoc(mt_result, &min_value, &max_value, &min_point, &max_point);

        tm->match(cc, ca, mt_result);
        cv::cuda::minMaxLoc(mt_result, &min_value, &max_value, &min_point, &max_point);

        tm->match(cd, ca, mt_result);
        cv::cuda::minMaxLoc(mt_result, &min_value, &max_value, &min_point, &max_point);

        gettimeofday(&end_time, NULL);
        timersub(&end_time, &start_time, &delta_times[i]);
        std::cout << "gpu: took: " << (delta_times[i].tv_usec + 1000000 * delta_times[i].tv_sec) / 1000. << std::endl;
        mean_delta_time += (delta_times[i].tv_usec + 1000000 * delta_times[i].tv_sec) / 1000.;

    }

    ga.release();
    gb.release();
    gc.release();
    gd.release();

    mean_delta_time /= N;
    std::cout << "gpu: mean time: " << mean_delta_time << std::endl;
    return mean_delta_time;
}


float time_cpu() {
    mean_delta_time = 0;
    cv::randu(a, cv::Scalar(0), cv::Scalar(100));
    cv::randu(b, cv::Scalar(0), cv::Scalar(100));
    cv::randu(c, cv::Scalar(0), cv::Scalar(100));
    cv::randu(d, cv::Scalar(0), cv::Scalar(100));

    cv::Rect tcrop(824, 824, 400, 400);
    cv::Rect mcrop(774, 774, 500, 500);

    cv::Mat mt_result(500-400+1, 500-400+1, CV_32F);

    for (int i=0; i < N; i++) {
        gettimeofday(&start_time, NULL);

        // crop images
        cv::Mat ca = a(tcrop);
        cv::Mat cb = b(mcrop);
        cv::Mat cc = c(mcrop);
        cv::Mat cd = d(mcrop);

        // match template with b, c, d
        cv::matchTemplate(cb, ca, mt_result, CV_TM_CCORR);
        cv::minMaxLoc(mt_result, &min_value, &max_value, &min_point, &max_point);

        cv::matchTemplate(cc, ca, mt_result, CV_TM_CCORR);
        cv::minMaxLoc(mt_result, &min_value, &max_value, &min_point, &max_point);

        cv::matchTemplate(cd, ca, mt_result, CV_TM_CCORR);
        cv::minMaxLoc(mt_result, &min_value, &max_value, &min_point, &max_point);


        gettimeofday(&end_time, NULL);
        timersub(&end_time, &start_time, &delta_times[i]);
        std::cout << "cpu: took: " << (delta_times[i].tv_usec + 1000000 * delta_times[i].tv_sec) / 1000. << std::endl;
        mean_delta_time += (delta_times[i].tv_usec + 1000000 * delta_times[i].tv_sec) / 1000.;
    }

    mean_delta_time /= N;
    std::cout << "cpu: mean time: " << mean_delta_time << std::endl;
    return mean_delta_time;
}

int main() {
   double cpu = time_cpu();
   double gpu = time_gpu();
   std::cout << std::endl;
   std::cout << "CPU : " << cpu << std::endl;
   std::cout << "GPU : " << gpu << std::endl;
}
