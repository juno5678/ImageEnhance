#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string.h>
#include <iostream>
#include "AGC_lib.h"


int main(int argc,char **argv)
{
    if(argc != 2)
    {
        printf("you have to type the one picture path\n");
        return 0;
    }
    std::string current_frame_path = argv[1];





    cv::waitKey(0);
    return 0;


}
