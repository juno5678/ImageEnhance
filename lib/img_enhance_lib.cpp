#include "img_enhance_lib.h"
#include "opencv2/core.hpp"
#include <cmath>

void AutoGammaCorrection(const cv::Mat &src, cv::Mat &dst, const cv::String windowName)
{
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
    printf(" %s's gamma : %3f\n",windowName.c_str(),mean[0]);

    cv::LUT(src, lut, dst);
    cv::imshow(windowName + "look up table",show_lut);
}

void AutoLinearTransformation(const cv::Mat &src, cv::Mat &dst, const cv::String windowName)
{
    const int channels = src.channels();
    const int type = src.type();
    assert( type==CV_8UC1 || type==CV_8UC3 );
    int size = 256;
    auto mean = cv::mean(src);
    if(channels == 3)
        mean[0] = (mean[0]+mean[1]+mean[2])/3;

    float range = 0.2;
    float first_corner_ratio = 1-range;
    float second_corner_ratio = 1+range;
    float first_corner_input = mean[0] * first_corner_ratio;
    float second_corner_input = mean[0] * second_corner_ratio;
    float first_corner_output = size * (0.5-range/2) - 1;
    float second_corner_output = size * (0.5+range/2) - 1;

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
            cv::circle(show_lut,cv::Point(i, size - binValue),1,cv::Scalar(125));
            cv::circle(show_lut,cv::Point(first_corner_input, size - first_corner_output),1,cv::Scalar(255));
            cv::circle(show_lut,cv::Point(second_corner_input, size - second_corner_output),1,cv::Scalar(255));
        }
    }
    else if(channels == 3)
    {
        for(int i = 0 ; i < size ; i++)
        {
            int Y = linearFormula(i,first_corner_input,second_corner_input,first_corner_output,second_corner_output);
            binValue = cv::saturate_cast<uchar>(Y);
            lut.at<cv::Vec3b>(0,i) = cv::Vec3b(binValue,binValue,binValue);

            cv::circle(show_lut,cv::Point(i, size - binValue),1,cv::Scalar(125));
            cv::circle(show_lut,cv::Point(first_corner_input, size - first_corner_output),1,cv::Scalar(255));
            cv::circle(show_lut,cv::Point(second_corner_input, size - second_corner_output),1,cv::Scalar(255));
        }
    }

    printf(" %s's linear transform : P1( %d , %d ) P2( %d , %d ) \n",windowName.c_str(),(int)first_corner_input,(int)first_corner_output,(int)second_corner_input,(int)second_corner_output);

    cv::LUT(src, lut, dst);
    cv::imshow(windowName + "look up table",show_lut);



}
//void AutoLinearTransformation(const cv::Mat &src, cv::Mat &dst, const cv::String windowName)
//{
//    const int channels = src.channels();
//    const int type = src.type();
//    assert( type==CV_8UC1 || type==CV_8UC3 );
//
//    int size = 256;
//
//    std::vector<std::vector<float>> pixel_pdf(channels,std::vector<float>(size,0));
//    std::vector<std::vector<float>> pixel_cdf(channels,std::vector<float>(size,0));
//    // count each pixel value number
//    for(int i = 0 ; i < src.rows; i ++)
//    {
//        for(int j = 0 ; j < src.cols ; j++)
//        {
//            for(int c = 0 ; c < channels ; c++)
//            {
//                int value = src.at<cv::Vec3b>(i,j)[c];
//                pixel_pdf.at(c).at(value) += 1;
//            }
//        }
//    }
//
//    // calculate pdf and cdf
//    // find two corner input value
//    float accumulation_p[channels] = {0};
//    float first_corner_input[channels] = {0};
//    float second_corner_input[channels] = {0};
//    bool find_first_corner[channels] = {false};
//    bool find_second_corner[channels] = {false};
//
//    float first_corner_ratio = 0.25;
//    float second_corner_ratio = 0.5;
//    for(int c = 0 ; c < channels ; c++)
//    {
//        for(int k = 0 ; k < size ; k++)
//        {
//            pixel_pdf.at(c).at(k) = pixel_pdf.at(c).at(k) / (src.rows*src.cols);
//            accumulation_p[c] += pixel_pdf.at(c).at(k);
//
//            pixel_cdf.at(c).at(k) = accumulation_p[c];
//
//            if(find_first_corner[c] == false && accumulation_p[c] >= first_corner_ratio)
//            {
//                first_corner_input[c] = k;
//                find_first_corner[c] = true;
//            }
//            if(find_second_corner[c] == false && accumulation_p[c] >= second_corner_ratio)
//            {
//                second_corner_input[c] = k;
//                find_second_corner[c] = true;
//            }
//        }
//    }
//    if(channels == 3)
//    {
//        first_corner_input[0] = (first_corner_input[0]+first_corner_input[1]+first_corner_input[2])/3;
//        second_corner_input[0] = (second_corner_input[0]+second_corner_input[1]+second_corner_input[2])/3;
//    }
//
//
//    float first_corner_output =  size * first_corner_ratio - 1;
//    float second_corner_output = size * second_corner_ratio - 1;
//    //float first_corner_output =  size * 0.25 - 1;
//    //float second_corner_output = size * 0.75 - 1;
//
//    // build look up table
//    cv::Mat lut(1,256,src.type());
//    cv::Mat show_lut = cv::Mat::zeros(size, size, CV_8UC1);
//
//    // use look up table to transform input
//    float binValue = 0;
//    if(channels == 1)
//    {
//        for(int i = 0 ; i < size ; i++)
//        {
//            int Y = linearFormula(i,first_corner_input[0],second_corner_input[0],first_corner_output,second_corner_output);
//            binValue = cv::saturate_cast<uchar>(Y);
//            lut.at<uchar>(0,i) = binValue;
//            cv::circle(show_lut,cv::Point(i, size - binValue),1,cv::Scalar(125));
//            cv::circle(show_lut,cv::Point(first_corner_input[0], size - first_corner_output),1,cv::Scalar(255));
//            cv::circle(show_lut,cv::Point(second_corner_input[0], size - second_corner_output),1,cv::Scalar(255));
//        }
//    }
//    else if(channels == 3)
//    {
//        for(int i = 0 ; i < size ; i++)
//        {
//            int Y = linearFormula(i,first_corner_input[0],second_corner_input[0],first_corner_output,second_corner_output);
//            binValue = cv::saturate_cast<uchar>(Y);
//            lut.at<cv::Vec3b>(0,i) = cv::Vec3b(binValue,binValue,binValue);
//
//            cv::circle(show_lut,cv::Point(i, size - binValue),1,cv::Scalar(125));
//            cv::circle(show_lut,cv::Point(first_corner_input[0], size - first_corner_output),1,cv::Scalar(255));
//            cv::circle(show_lut,cv::Point(second_corner_input[0], size - second_corner_output),1,cv::Scalar(255));
//        }
//    }
//
//    printf(" %s's linear transform : P1( %d , %d ) P2( %d , %d ) \n",windowName.c_str(),(int)first_corner_input[0],(int)first_corner_output,(int)second_corner_input[0],(int)second_corner_output);
//
//    cv::LUT(src, lut, dst);
//    cv::imshow(windowName + "look up table",show_lut);
//}
int linearFormula(int input,float first_corner_input,float second_corner_input,float first_corner_output,float second_corner_output)
{
    int Y = 0;
    if(input < first_corner_input)
        Y = std::round(first_corner_output / first_corner_input * input);
    else if( input >= first_corner_input && input < second_corner_input)
    {
        float m = (second_corner_output - first_corner_output) / (second_corner_input - first_corner_input);
        float b = first_corner_output - m * first_corner_input ;
        Y =  std::round(m * input + b) ;
        //printf(" m : %3f , b : %3f \t",m,b);
        //printf(" linar transform ( %d , %d )\n",input,Y);
    }
    else if(input >= second_corner_input)
    {
        float m = (255 - second_corner_output) / (255 - second_corner_input);
        float b = second_corner_output - m * second_corner_input ;
        Y =  std::round(m * input + b) ;
        printf(" m : %3f , b : %3f \t",m,b);
        printf(" linar transform ( %d , %d )\n",input,Y);
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
                if(windowName == "src_hist")
                {
                    if(i == 30 || i == 250)
                        cv::line(dstImage,cv::Point(j*size+i, size - 1),cv::Point(j*size+i, size - realValue),cv::Scalar(255,255,0));
                }
                else if(windowName == "dst_hist")
                {
                    if(i == 83 || i == 250)
                        cv::line(dstImage,cv::Point(j*size+i, size - 1),cv::Point(j*size+i, size - realValue),cv::Scalar(255,255,0));
                }

            }

        }

    }

    cv::imshow(windowName, dstImage);
}
