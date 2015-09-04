#include <time.h>
#include <sys/time.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/cudaarithm.hpp"

#define N 10


//using namespace std;
//using namespace cv;


struct timeval start_time, end_time;
struct timeval delta_times[N];
double mean_delta_time;

cv::Mat a(2048, 2048, CV_32F, cv::Scalar(0.));
cv::Mat b(2048, 2048, CV_32F, cv::Scalar(0.));


double time_gpu_upload() {
    mean_delta_time = 0;
    cv::randu(a, cv::Scalar(0), cv::Scalar(100));
    
    cv::cuda::GpuMat ga(2048, 2048, CV_32F);


    // transfer both the gpu
    for (int i=0; i < N; i++) {
        gettimeofday(&start_time, NULL);

        ga.upload(a);

        gettimeofday(&end_time, NULL);
        timersub(&end_time, &start_time, &delta_times[i]);
        std::cout << "gpu: upload took: " << (delta_times[i].tv_usec + 1000000 * delta_times[i].tv_sec) / 1000. << std::endl;
        mean_delta_time += (delta_times[i].tv_usec + 1000000 * delta_times[i].tv_sec) / 1000.;

    }

    ga.release();

    mean_delta_time /= N;
    std::cout << "gpu: mean time: " << mean_delta_time << std::endl;
    return mean_delta_time;
}


double time_gpu_download() {
    mean_delta_time = 0;
    cv::randu(a, cv::Scalar(0), cv::Scalar(100));
    cv::randu(b, cv::Scalar(0), cv::Scalar(100));
    
    cv::cuda::GpuMat ga(2048, 2048, CV_32F);
    cv::cuda::GpuMat gb(2048, 2048, CV_32F);

    ga.upload(a);
    gb.upload(b);

    cv::cuda::multiply(ga, gb, gb);

    ga.release();

    // transfer both the gpu
    for (int i=0; i < N; i++) {
        gettimeofday(&start_time, NULL);

        gb.download(b);

        gettimeofday(&end_time, NULL);
        timersub(&end_time, &start_time, &delta_times[i]);
        std::cout << "gpu: upload took: " << (delta_times[i].tv_usec + 1000000 * delta_times[i].tv_sec) / 1000. << std::endl;
        mean_delta_time += (delta_times[i].tv_usec + 1000000 * delta_times[i].tv_sec) / 1000.;

    }

    gb.release();

    mean_delta_time /= N;
    std::cout << "gpu: mean time: " << mean_delta_time << std::endl;
    return mean_delta_time;
}


double time_gpu_multiply() {
    mean_delta_time = 0;
    cv::randu(a, cv::Scalar(0), cv::Scalar(100));
    cv::randu(b, cv::Scalar(0), cv::Scalar(100));
    
    cv::cuda::GpuMat ga(2048, 2048, CV_32F);
    cv::cuda::GpuMat gb(2048, 2048, CV_32F);

    ga.upload(a);
    gb.upload(b);

    // transfer both the gpu
    for (int i=0; i < N; i++) {
        gettimeofday(&start_time, NULL);

        cv::cuda::multiply(ga, gb, gb);

        gettimeofday(&end_time, NULL);
        timersub(&end_time, &start_time, &delta_times[i]);
        std::cout << "gpu: multiply took: " << (delta_times[i].tv_usec + 1000000 * delta_times[i].tv_sec) / 1000. << std::endl;
        mean_delta_time += (delta_times[i].tv_usec + 1000000 * delta_times[i].tv_sec) / 1000.;

    }

    mean_delta_time /= N;
    std::cout << "gpu: mean time: " << mean_delta_time << std::endl;
    return mean_delta_time;
}


float time_cpu_multiply() {
    mean_delta_time = 0;
    cv::randu(a, cv::Scalar(0), cv::Scalar(100));
    cv::randu(b, cv::Scalar(0), cv::Scalar(100));

    for (int i=0; i < N; i++) {
        gettimeofday(&start_time, NULL);

        cv::multiply(a, b, b);

        gettimeofday(&end_time, NULL);
        timersub(&end_time, &start_time, &delta_times[i]);
        std::cout << "cpu: multiply took: " << (delta_times[i].tv_usec + 1000000 * delta_times[i].tv_sec) / 1000. << std::endl;
        mean_delta_time += (delta_times[i].tv_usec + 1000000 * delta_times[i].tv_sec) / 1000.;
    }

    mean_delta_time /= N;
    std::cout << "cpu: mean time: " << mean_delta_time << std::endl;
    return mean_delta_time;
}

int main() {
   double cpu_multiply = time_cpu_multiply();
   double gpu_upload = time_gpu_upload();
   double gpu_multiply = time_gpu_multiply();
   double gpu_download = time_gpu_download();
   std::cout << std::endl;
   std::cout << "CPU multiply: " << cpu_multiply << std::endl;
   std::cout << "GPU upload  : " << gpu_upload << std::endl;
   std::cout << "GPU download: " << gpu_download << std::endl;
   std::cout << "GPU multiply: " << gpu_multiply << std::endl;
}
