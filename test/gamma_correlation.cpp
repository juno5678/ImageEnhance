#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string.h>
#include <iostream>
#include "img_enhance_lib.h"


int main(int argc,char **argv)
{
    if(argc != 2)
    {
        printf("you have to type the one picture path\n");
        return 0;
    }
    std::string input_image_path = argv[1];
    cv::Mat src = cv::imread(input_image_path);

    if(src.empty())
    {
        printf("%s didm't have picture\n",input_image_path.c_str());
        return 0;
    }

    cv::Mat gray;
    cv::cvtColor(src,gray,cv::COLOR_BGR2GRAY);
    cv::Mat dst;
    cv::Mat gray_dst;

    AutoGammaCorrection(src,dst,"color");
    AutoGammaCorrection(gray,gray_dst,"gray");
    draw_hist(src,"src_hist");
    draw_hist(dst,"dst_hist");
    draw_hist(gray,"gray_hist");
    draw_hist(gray_dst,"grat_dst_hist");

    cv::imshow("src",src);
    cv::imshow("gray",gray);
    cv::imshow("gray dst",gray_dst);
    cv::imshow("dst",dst);

    cv::waitKey(0);
    return 0;


}
