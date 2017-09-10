#include <opencv2\highgui\highgui.hpp>
#include <opencv2\core\core.hpp>
#include <opencv2\opencv.hpp>
#include<iostream>
#include<vector>
#include <stdlib.h>

#include "interval_gradient.hpp"
#include "GaussianBlur.hpp"
int main() {
	//load image
	//in the interval_filtering\interval_filtering, I put some images there.
	cv::Mat image = cv::imread("super.jpg", 1);
	if (!image.data) {
		printf("Error loading src \n");
		return -1;
	}
	
	/*cv::Mat a(cv::Size(2,3), CV_32FC3, cv::Scalar(3, 2, 1));
	cv::Mat b(cv::Size(2,3), CV_32FC3, cv::Scalar::all(2));
	std::cout << a-b;*/
	int sigma = 3; 
	cv::Mat I;
	image.convertTo(I, CV_32FC3);//convert to single
	I *= 1.0 / 255.0;
	cv::Mat pic = interval_Filter(I, sigma);

	cv::namedWindow("Input Image", CV_WINDOW_FREERATIO);
	imshow("Input Image", image);

	cv::waitKey(0);

}