#include "img_enhance_lib.h"
#include "opencv2/core.hpp"
#include <cmath>

void AutoGammaCorrection(const cv::Mat &src, cv::Mat &dst, const cv::String windowName)
{
    printf("----------------------------start gamma correction------------------------\n");
    const int channels = src.channels();
    const int type = src.type();
    assert( type==CV_8UC1 || type==CV_8UC3 );


    auto mean = cv::mean(src);
    mean[0] = std::log10(0.5) / std::log10(mean[0]/255); // gamma = -0.3/log10(X)
    if(channels == 3)
    {
        mean[1] = std::log10(0.5) / std::log10(mean[1]/255);
        mean[2] = std::log10(0.5) / std::log10(mean[2]/255);

        auto mean3 = (mean[0]+mean[1]+mean[2])/3;
        mean[0]=mean[1]=mean[2] = mean3;
    }

    // build look up table

    int size = 256;
    cv::Mat lut(1,256,src.type());
    cv::Mat show_lut = cv::Mat::zeros(size, size, CV_8UC1);
    if(channels == 1)
    {
        for(int i = 0 ; i < 256 ; i++)
        {
            // normalized
            float Y = i*1.0f/255;
            // the value after gamma correlation
            Y = std::pow(Y,mean[0]);

            lut.at<uchar>(0,i) = cv::saturate_cast<uchar>(Y*255);

            float binValue = lut.at<uchar>(0,i);
            cv::circle(show_lut,cv::Point(i, size - binValue),1,cv::Scalar(255));
        }
    }
    else if(channels == 3)
    {
        for(int i=0 ; i < 256 ; i++)
        {
            float Y = i*1.0f/255;
            Y = std::pow(Y,mean[0]);

            auto value =  cv::saturate_cast<uchar>(Y*255);

            lut.at<cv::Vec3b>(0,i) = cv::Vec3b(value,value,value);

            float binValue = lut.at<cv::Vec3b>(0,i)[0];
            cv::circle(show_lut,cv::Point(i, size - binValue),1,cv::Scalar(255));
        }
    }
    printf("%s's gamma : %3f\n",windowName.c_str(),mean[0]);

    cv::LUT(src, lut, dst);

    cv::Scalar inputMean, outputMean;
    cv::Scalar inputStdDev, outputStdDev;
    cv::meanStdDev(src, inputMean, inputStdDev);
    cv::meanStdDev(dst, outputMean, outputStdDev);
    printf("input mean : %1f , input stddev : %1f \n",inputMean[0],inputStdDev[0]);
    printf("output mean : %1f , output stddev : %1f \n",outputMean[0], outputStdDev[0]);

    cv::imshow(windowName + "look up table",show_lut);
}

void AutoLinearTransformation(const cv::Mat &src, cv::Mat &dst, const cv::String windowName)
{
    printf("----------------------------start linear transformation------------------------\n");
    const int channels = src.channels();
    const int type = src.type();
    assert( type==CV_8UC1 || type==CV_8UC3 );
    int size = 256;

    cv::Mat expectArray(1,size,CV_32FC1);
    cv::Scalar expectMean, inputMean, outputMean;
    cv::Scalar expectStdDev, inputStdDev, outputStdDev;
    for(int i = 0 ; i < size ; i++ )
        expectArray.at<float>(0,i) = i;

    cv::meanStdDev(expectArray, expectMean, expectStdDev);
    cv::meanStdDev(src, inputMean, inputStdDev);

    if(channels == 3)
    {
        inputMean[0] = (inputMean[0]+inputMean[1]+inputMean[2])/3;
        inputStdDev[0] = (inputStdDev[0]+inputStdDev[1]+inputStdDev[2])/3;
    }

    float first_corner_input   = cv::saturate_cast<uchar>(inputMean[0] - inputStdDev[0]);
    float second_corner_input  = cv::saturate_cast<uchar>(inputMean[0] + inputStdDev[0]);
    float first_corner_output  = cv::saturate_cast<uchar>(expectMean[0] - expectStdDev[0]);
    float second_corner_output = cv::saturate_cast<uchar>(expectMean[0] + expectStdDev[0]);

    printf("input mean : %1f , input stddev : %1f \n",inputMean[0],inputStdDev[0]);
    printf("expect mean : %1f , expect stddev : %1f \n",expectMean[0], expectStdDev[0]);
    // build look up table
    cv::Mat lut(1,256,src.type());
    cv::Mat show_lut = cv::Mat::zeros(size, size, CV_8UC1);

    // use look up table to transform input
    float binValue = 0;
    if(channels == 1)
    {
        for(int i = 0 ; i < size ; i++)
        {
            int Y = linearFormula(i,first_corner_input,second_corner_input,first_corner_output,second_corner_output);
            binValue = cv::saturate_cast<uchar>(Y);
            lut.at<uchar>(0,i) = binValue;
        }
    }
    else if(channels == 3)
    {
        for(int i = 0 ; i < size ; i++)
        {
            int Y = linearFormula(i,first_corner_input,second_corner_input,first_corner_output,second_corner_output);
            binValue = cv::saturate_cast<uchar>(Y);
            lut.at<cv::Vec3b>(0,i) = cv::Vec3b(binValue,binValue,binValue);
        }
    }
    cv::line(show_lut,cv::Point(0,size),cv::Point(first_corner_input, size - first_corner_output),cv::Scalar(127),1);
    cv::line(show_lut,cv::Point(first_corner_input, size - first_corner_output),cv::Point(second_corner_input, size - second_corner_output),cv::Scalar(127),1);
    cv::line(show_lut,cv::Point(second_corner_input, size - second_corner_output),cv::Point(size,0),cv::Scalar(127),1);
    cv::circle(show_lut,cv::Point(first_corner_input, size - first_corner_output),1,cv::Scalar(255));
    cv::circle(show_lut,cv::Point(second_corner_input, size - second_corner_output),1,cv::Scalar(255));



    cv::LUT(src, lut, dst);

    cv::meanStdDev(dst, outputMean, outputStdDev);
    printf("output mean : %1f , output stddev : %1f \n",outputMean[0], outputStdDev[0]);
    printf("%s's linear transform : P1( %d , %d ) P2( %d , %d ) \n",windowName.c_str(),(int)first_corner_input,(int)first_corner_output,(int)second_corner_input,(int)second_corner_output);

    cv::imshow(windowName + " look up table",show_lut);

}
int linearFormula(int input,float first_corner_input,float second_corner_input,float first_corner_output,float second_corner_output)
{
    int Y = 0;
    float epsilon = 0.001;
    if(input < first_corner_input)
        Y = std::round(first_corner_output / first_corner_input * input + epsilon);
    else if( input >= first_corner_input && input < second_corner_input)
    {
        float m = (second_corner_output - first_corner_output) / (second_corner_input - first_corner_input +epsilon);
        float b = first_corner_output - m * first_corner_input ;
        Y =  std::round(m * input + b) ;
        //printf(" m : %3f , b : %3f \t",m,b);
        //printf(" linar transform ( %d , %d )\n",input,Y);
    }
    else if(input >= second_corner_input)
    {
        float m = (255 - second_corner_output) / (255 - second_corner_input + epsilon);
        float b = second_corner_output - m * second_corner_input ;
        Y =  std::round(m * input + b) ;
        //printf(" m : %3f , b : %3f \t",m,b);
        //printf(" linar transform ( %d , %d )\n",input,Y);
    }
    return Y;
}
void draw_hist(cv::Mat &src, const cv::String windowName)
{
    int channels = src.channels();
    const int type = src.type();
    assert( type==CV_8UC1 || type==CV_8UC3 );
    std::vector<cv::Mat> dstHist(channels);

    float range[] = { 0, 256 };
    const float *ranges[] = { range };
    int size = 256;

    double maxValue[channels];
    for(int i = 0 ; i < channels ; i++)
    {
        cv::calcHist(&src, 1, &i, cv::Mat(), dstHist[i], 1, &size, ranges);
        cv::minMaxLoc(dstHist[i], 0, &maxValue[i], 0, 0);
    }

    cv::Mat dstImage = cv::Mat::zeros(size, size*channels, CV_8UC3);

    int hpt = cv::saturate_cast<int>(0.9 * size);
    cv::Scalar color[4] = {cv::Scalar(255,255,255),cv::Scalar(255,0,0),cv::Scalar(0,255,0),cv::Scalar(0,0,255)};
    for(int i = 0 ; i < 256 ; i++ )
    {
        for(int j = 0 ; j < channels ; j++)
        {
            float binValue = dstHist[j].at<float>(i);
            int realValue = cv::saturate_cast<int>(binValue * hpt/maxValue[j]);
            if(channels == 1)
                cv::line(dstImage,cv::Point(i, size - 1),cv::Point(i, size - realValue),color[0]);
            else if(channels == 3)
            {
                cv::line(dstImage,cv::Point(j*size+i, size - 1),cv::Point(j*size+i, size - realValue),color[j+1]);

            }

        }

    }

    cv::imshow(windowName, dstImage);
}
