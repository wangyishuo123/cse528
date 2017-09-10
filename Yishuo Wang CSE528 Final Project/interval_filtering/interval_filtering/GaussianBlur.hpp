#ifndef GAUSSIANBLUR_HPP
#define GAUSSIANBLUR_HPP

#include <opencv2\highgui\highgui.hpp>
#include <opencv2\core\core.hpp>
#include <opencv2\opencv.hpp>

#define PI 3.1415926
cv::Mat gauss2(cv::Mat input, int sigma);
void gauss3(cv::InputArray _src, cv::OutputArray _dst, cv::Size ksize, double sigma1, double sigma2, int borderType);

cv::Mat gauss(cv::Mat input, int sigma) {
	int fr = ceil(3 * sigma);

	cv::Mat output;
	//gauss2(input, sigma);

	//cv::GaussianBlur(input, output, cv::Size(2 * fr + 1, 1), sigma, sigma, cv::BORDER_REFLECT);
	gauss3(input, output, cv::Size(2 * fr + 1, 1), sigma, sigma, cv::BORDER_REFLECT);

	return output;
}

cv::Mat gauss2(cv::Mat input, int sigma) {
	int fr = ceil(3 * sigma);
	cv::Mat result(cv::Size(input.size()), CV_32FC3, cv::Scalar(0, 0, 0));
	cv::Mat weight(cv::Size(2 * fr + 1, 1), CV_32FC1, cv::Scalar(0));
	int index = 0;
	for (int i = -1 * fr; i <= fr; i++){	
		weight.at<float>(0, index) = float(1) / (float)(sigma * std::sqrt(2 * PI)) * std::exp(-1 * i * i / (2 * sigma * sigma));
	}

	cv::Scalar dx = cv::sum(weight);
	float v = dx[0];
	weight = weight / v;
	
	for (int i = 0; i < input.rows; i++) {
		for (int j = 0; j < input.cols; j++) {
			cv::Mat w(cv::Size(2 * fr + 1, 1), CV_32FC3, cv::Scalar(0,0,0));
			for (int k = -1 * fr; k <= fr; k++){
				int c = j + k;
				if (c < 0){
					c = -1 * c - 1;
				}
				else if(c >= input.cols){
					c = c - input.cols;
					c = input.cols - 1 - c;
				}
				w.at<cv::Vec3f>(0, k + fr) = input.at<cv::Vec3f>(i, c) * weight.at<float>(0, k + fr);
			}
			cv::Scalar d = cv::sum(w);
			result.at<cv::Vec3f>(i, j)[0] = d[0];
			result.at<cv::Vec3f>(i, j)[1] = d[1];
			result.at<cv::Vec3f>(i, j)[2] = d[2];
		}
	}
	return result;
}
//gauss3 get from http://www.voidcn.com/blog/sulanqing/article/p-2233820.html and the theory is http://www.pixelstech.net/article/1353768112-Gaussian-Blur-Algorithm
//and did a little change
void gauss3(cv::InputArray _src, cv::OutputArray _dst, cv::Size ksize, double sigma1, double sigma2, int borderType)
{
	cv::Mat src = _src.getMat();
	_dst.create(src.size(), src.type());
	cv::Mat dst = _dst.getMat();

	cv::Ptr<cv::FilterEngine> f = cv::createGaussianFilter(src.type(), ksize, sigma1, sigma2, borderType);
	f->apply(src, dst);
}

#endif