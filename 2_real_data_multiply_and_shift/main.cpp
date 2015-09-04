#include <time.h>
#include <sys/time.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudaimgproc.hpp"

//#define NORMED
#ifdef NORMED
    #define METHOD CV_TM_CCORR_NORMED
    //#define METHOD CV_TM_SQDIFF_NORMED
#else
    #define METHOD CV_TM_CCORR
    //#define METHOD CV_TM_SQDIFF
#endif

struct timeval start_time, end_time;
struct timeval delta_time;
double mean_delta_time;

double min_value, max_value;
cv::Point min_point, max_point;

cv::Mat im0, im1, im2, im3, imm;
cv::Mat fim0, fim1, fim2, fim3, fimm;
cv::Mat nfim0, nfim1, nfim2, nfim3;
/*
cv::Mat fim0(2048, 2048, CV_32F, cv::Scalar(0.));
cv::Mat fim1(2048, 2048, CV_32F, cv::Scalar(0.));
cv::Mat fim2(2048, 2048, CV_32F, cv::Scalar(0.));
cv::Mat fim3(2048, 2048, CV_32F, cv::Scalar(0.));
cv::Mat fimm(2048, 2048, CV_32F, cv::Scalar(0.));
*/
cv::cuda::GpuMat gim0, gim1, gim2, gim3, gimm;
cv::cuda::GpuMat ngim0, ngim1, ngim2, ngim3;

cv::Ptr<cv::cuda::TemplateMatching> tm = cv::cuda::createTemplateMatching(CV_32F, METHOD);
cv::Rect tcrop(824, 824, 400, 400);
cv::Rect mcrop(774, 774, 500, 500);
cv::cuda::GpuMat cgim0(400, 400, CV_32F);
cv::cuda::GpuMat cgim1(500, 500, CV_32F);
cv::cuda::GpuMat cgim2(500, 500, CV_32F);
cv::cuda::GpuMat cgim3(500, 500, CV_32F);
cv::cuda::GpuMat mt_result_0(500-400+1, 500-400+1, CV_32F);
cv::cuda::GpuMat mt_result_1(500-400+1, 500-400+1, CV_32F);
cv::cuda::GpuMat mt_result_2(500-400+1, 500-400+1, CV_32F);

cv::cuda::Stream gpu_stream;

void print_device_info() {
    cv::cuda::DeviceInfo di = cv::cuda::getDevice();
    std::cout << "Device info:" << std::endl;
    std::cout << "\t" << di.freeMemory() << std::endl;
    std::cout << "\t" << di.totalMemory() << std::endl;
    //std::cout << "\t" << std::endl;
}

void load_images() {
    im0 = cv::imread("../data/0.tif");
    im1 = cv::imread("../data/1.tif");
    im2 = cv::imread("../data/2.tif");
    im3 = cv::imread("../data/3.tif");
    imm = cv::imread("../data/m.tif");

    /*
    fim0 = cv::Mat(2048, 2048, CV_32F);
    fim1 = cv::Mat(2048, 2048, CV_32F);
    fim2 = cv::Mat(2048, 2048, CV_32F);
    fim3 = cv::Mat(2048, 2048, CV_32F);
    fimm = cv::Mat(2048, 2048, CV_32F);
    */

    // std::cout << "Mean size: " << imm.size() << " type: " << imm.type() << " depth: " << imm.depth() << std::endl;

    // process mean
    double minv, maxv;
    cv::minMaxLoc(imm, &minv, &maxv);
    // std::cout << "Mean (after load) min: " << minv << " max: " << maxv << std::endl;
    imm.convertTo(fimm, CV_32F);
    cv::Scalar mm = cv::mean(fimm);
    cv::divide(mm, fimm, fimm);
    cv::minMaxLoc(fimm, &minv, &maxv);
    // std::cout << "Mean (after norm) min: " << minv << " max: " << maxv << std::endl;
}

void convert_image_types() {
    im0.convertTo(fim0, CV_32F);
    im1.convertTo(fim1, CV_32F);
    im2.convertTo(fim2, CV_32F);
    im3.convertTo(fim3, CV_32F);
}

void upload_images() {
    gim0.upload(fim0, gpu_stream);
    gim1.upload(fim1, gpu_stream);
    gim2.upload(fim2, gpu_stream);
    gim3.upload(fim3, gpu_stream);
    gimm.upload(fimm, gpu_stream);
}

void normalize_images() {
    cv::cuda::multiply(gimm, gim0, ngim0, 1., -1, gpu_stream);
    cv::cuda::multiply(gimm, gim1, ngim1, 1., -1, gpu_stream);
    cv::cuda::multiply(gimm, gim2, ngim2, 1., -1, gpu_stream);
    cv::cuda::multiply(gimm, gim3, ngim3, 1., -1, gpu_stream);
}

void find_shift() {

}

void download_results() {
    gim0.download(fim0, gpu_stream);
    gim1.download(fim1, gpu_stream);
    gim2.download(fim2, gpu_stream);
    gim3.download(fim3, gpu_stream);

    ngim0.download(fim0, gpu_stream);
    ngim1.download(fim1, gpu_stream);
    ngim2.download(fim2, gpu_stream);
    ngim3.download(fim3, gpu_stream);

}

void release_images() {
    im0.release();
    im1.release();
    im2.release();
    im3.release();

    fim0.release();
    fim1.release();
    fim2.release();
    fim3.release();

    gim0.release();
    gim1.release();
    gim2.release();
    gim3.release();

    imm.release();
    fimm.release();
    gimm.release();
} 

void timer_start() {
    gettimeofday(&start_time, NULL);
}

void timer_end() {
    gettimeofday(&end_time, NULL);
    timersub(&end_time, &start_time, &delta_time);
}

double timer_ms() {
    return (delta_time.tv_usec + 1000000 * delta_time.tv_sec) / 1000.;
}

int main() {
    print_device_info();

    load_images();

    timer_start();
    convert_image_types();  // Weirdly slow: ~27 ms per conversion?
    timer_end();
    std::cout << "convert_image_types: " << timer_ms() << std::endl;

    timer_start();
    upload_images();
    gpu_stream.waitForCompletion();
    timer_end();
    std::cout << "upload_images: " << timer_ms() << std::endl;

    timer_start();
    normalize_images();
    gpu_stream.waitForCompletion();
    timer_end();
    std::cout << "normalize_images: " << timer_ms() << std::endl;

    timer_start();
    download_results();
    gpu_stream.waitForCompletion();
    timer_end();
    std::cout << "download_results: " << timer_ms() << std::endl;

    release_images();
    //cv::cuda::resetDevice();
}
