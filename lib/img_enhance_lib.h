#ifndef __IMG_ENHANCE_LIB_H__
#define __IMG_ENHANCE_LIB_H__

#include <iostream>
#include <math.h>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <time.h>
#include <chrono>
#include <vector>

void AutoGammaCorrection(const cv::Mat &src, cv::Mat &dst, const cv::String windowName);
void AutoLinearTransformation(const cv::Mat &src, cv::Mat &dst, const cv::String windowName);
void draw_hist(cv::Mat &src, const cv::String windowName);
int linearFormula(int input,float first_corner_input,float second_corner_input,float first_corner_output,float second_corner_output);


#endif
