#ifndef REBUILDE_HPP
#define REBUILDE_HPP

#include <opencv2\highgui\highgui.hpp>
#include <opencv2\core\core.hpp>
#include <opencv2\opencv.hpp>
#include <iostream>
#include <vector>
#include <stdlib.h>

#include "rescaled.hpp"
#include "GaussianBlur.hpp"
cv::Mat CalcRGBmax(cv::Mat i_RGB);

cv::Mat reconstruct_1d(cv::Mat image, cv::Mat rescaled, int sigma, float epsi) {
/**************************************************************************************************************/
/*                          1 Rp                                                                              */
/**************************************************************************************************************/
	//1 build Rp
	cv::Mat Rp(cv::Size(rescaled.size()), CV_32FC3, cv::Scalar(0, 0, 0));
	for (int i = 0; i < Rp.rows; i++) {
		Rp.at<cv::Vec3f>(i, 0) = image.at<cv::Vec3f>(i, 0);
		Rp.at<cv::Vec3f>(i, 1) = image.at<cv::Vec3f>(i, 0) + rescaled.at<cv::Vec3f>(i, 0);
		for (int j = 2; j < Rp.cols; j++) {
			Rp.at<cv::Vec3f>(i, j) = Rp.at<cv::Vec3f>(i, j - 1) + rescaled.at<cv::Vec3f>(i, j - 1);	
		}
	}
	//get g(Rp)
	cv::Mat Rp_out = gauss(Rp, sigma);

	//std::cout << "RG" << std::endl;
	//for (int i = 10; i < Rp_out.rows; i++) {//row
	//	for (int j = 0; j < Rp_out.cols; j++) {//col
	//		std::cout << Rp_out.at<float>(i, j) << std::endl;
	//	}
	//}
	//std::cout << "RG" << std::endl;
/**************************************************************************************************************/
/*                          2 Ip                                                                             */
/**************************************************************************************************************/
	//2 Ip is image. get g(Ip)
	cv::Mat Ip_out = gauss(image, sigma);


/**************************************************************************************************************/
/*                          3 RpIp                                                                    */
/**************************************************************************************************************/
	//3 get g(RpIp)
	cv::Mat RpIp_out(cv::Size(image.size()), CV_32FC3, cv::Scalar(0, 0, 0));
	cv::multiply(Rp, image, RpIp_out, 1.0);
	/*for (int i = 0; i < RpIp_out.rows; i++) {
		for (int j = 0; j < RpIp_out.cols; j++) {
			RpIp_out.at<cv::Vec3f>(i, j)[0] = Rp.at<cv::Vec3f>(i, j)[0] * image.at<cv::Vec3f>(i, j)[0];
			RpIp_out.at<cv::Vec3f>(i, j)[1] = Rp.at<cv::Vec3f>(i, j)[1] * image.at<cv::Vec3f>(i, j)[1];
			RpIp_out.at<cv::Vec3f>(i, j)[2] = Rp.at<cv::Vec3f>(i, j)[2] * image.at<cv::Vec3f>(i, j)[2];
		}
	}*/
	RpIp_out = gauss(RpIp_out, sigma);
	
/**************************************************************************************************************/
/*                          4 RpRp                                                                              */
/**************************************************************************************************************/
	//4 get g(RpRp)
	cv::Mat RpRp_out(cv::Size(image.size()), CV_32FC3, cv::Scalar(0, 0, 0));
	cv::pow(Rp, 2, RpRp_out);
	/*for (int i = 0; i < RpRp_out.rows; i++) {
		for (int j = 0; j < RpRp_out.cols; j++) {
			RpRp_out.at<cv::Vec3f>(i, j)[0] = pow(Rp.at<cv::Vec3f>(i, j)[0], 2);
			RpRp_out.at<cv::Vec3f>(i, j)[1] = pow(Rp.at<cv::Vec3f>(i, j)[1], 2);
			RpRp_out.at<cv::Vec3f>(i, j)[2] = pow(Rp.at<cv::Vec3f>(i, j)[2], 2);
		}
	}*/
	RpRp_out = gauss(RpRp_out, sigma);
	
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	cv::Mat a(cv::Size(image.size()), CV_32FC3, cv::Scalar(0, 0, 0));
	cv::Mat b(cv::Size(image.size()), CV_32FC3, cv::Scalar(0, 0, 0));

	cv::Mat top;
	cv::multiply(Rp_out, Ip_out, top, 1.0);
	top = RpIp_out - top;

	cv::Mat bot;
	cv::pow(Rp_out, 2, bot);
	bot = RpRp_out - bot;

	top = max(top, 0);
	bot = max(bot, 0);

	cv::Mat ep(cv::Size(image.size()), CV_32FC3, cv::Scalar(epsi, epsi, epsi));
	bot = bot + ep;
	cv::divide(top, bot, a, 1.0);

	cv::multiply(a, Rp_out, b, 1.0);
	b = Ip_out - b;

	//for (int i = 0; i < image.rows; i++) {
	//	for (int j = 0; j < image.cols; j++) {
	//		float temp = 0;
	//		cv::Mat top(cv::Size(1, 1), CV_32FC3, cv::Scalar(0, 0, 0));
	//		cv::Mat bot(cv::Size(1, 1), CV_32FC3, cv::Scalar(0, 0, 0));
	//		top.at<cv::Vec3f>(0, 0)[0] = RpIp_out.at<cv::Vec3f>(i, j)[0] 
	//			- Rp_out.at<cv::Vec3f>(i, j)[0] * Ip_out.at<cv::Vec3f>(i, j)[0];
	//		top.at<cv::Vec3f>(0, 0)[1] = RpIp_out.at<cv::Vec3f>(i, j)[1]
	//			- Rp_out.at<cv::Vec3f>(i, j)[1] * Ip_out.at<cv::Vec3f>(i, j)[1];
	//		top.at<cv::Vec3f>(0, 0)[2] = RpIp_out.at<cv::Vec3f>(i, j)[2]
	//			- Rp_out.at<cv::Vec3f>(i, j)[2] * Ip_out.at<cv::Vec3f>(i, j)[2];

	//		bot.at<cv::Vec3f>(0, 0)[0] = RpRp_out.at<cv::Vec3f>(i, j)[0] 
	//			- pow(Rp_out.at<cv::Vec3f>(i, j)[0], 2);
	//		bot.at<cv::Vec3f>(0, 0)[1] = RpRp_out.at<cv::Vec3f>(i, j)[1]
	//			- pow(Rp_out.at<cv::Vec3f>(i, j)[1], 2);
	//		bot.at<cv::Vec3f>(0, 0)[2] = RpRp_out.at<cv::Vec3f>(i, j)[2]
	//			- pow(Rp_out.at<cv::Vec3f>(i, j)[2], 2);
	//		top = max(top, 0);
	//		bot = max(bot, 0);
	//		/*if (i == 1)
	//			std::cout << top << std::endl;*/

	//		a.at<cv::Vec3f>(i, j)[0] = top.at<cv::Vec3f>(0, 0)[0] 
	//			/ (bot.at<cv::Vec3f>(0, 0)[0] + epsi);
	//		a.at<cv::Vec3f>(i, j)[1] = top.at<cv::Vec3f>(0, 0)[1]
	//			/ (bot.at<cv::Vec3f>(0, 0)[1] + epsi);
	//		a.at<cv::Vec3f>(i, j)[2] = top.at<cv::Vec3f>(0, 0)[2]
	//			/ (bot.at<cv::Vec3f>(0, 0)[2] + epsi);

	//		b.at<cv::Vec3f>(i, j)[0] = Ip_out.at<cv::Vec3f>(i, j)[0] - a.at<cv::Vec3f>(i, j)[0] * Rp_out.at<cv::Vec3f>(i, j)[0];
	//		b.at<cv::Vec3f>(i, j)[1] = Ip_out.at<cv::Vec3f>(i, j)[1] - a.at<cv::Vec3f>(i, j)[1] * Rp_out.at<cv::Vec3f>(i, j)[1];
	//		b.at<cv::Vec3f>(i, j)[2] = Ip_out.at<cv::Vec3f>(i, j)[2] - a.at<cv::Vec3f>(i, j)[2] * Rp_out.at<cv::Vec3f>(i, j)[2];
	//	}
	//}
	
	

	//prevent bleeding
	cv::Mat max_a = CalcRGBmax(a);
	//a = max(max_a, a);
	a = gauss(a, sigma);
	b = gauss(b, sigma);


/**************************************************************************************************************/
/*                   now a is g(a), b is g(b) -> S = g(a)*Rp + g(b)                                           */
/**************************************************************************************************************/
	//
	//S = g(a)*Rp + g(b)
	cv::Mat S(cv::Size(Rp.size()), CV_32FC3, cv::Scalar(0, 0, 0));
	cv::multiply(a, Rp, S, 1.0);
	S = S + b;
	/*for (int i = 0; i < S.rows; i++) {
		for (int j = 0; j < S.cols; j++) {
			S.at<cv::Vec3f>(i, j)[0] = a.at<cv::Vec3f>(i, j)[0] * Rp.at<cv::Vec3f>(i, j)[0] + b.at<cv::Vec3f>(i, j)[0];
			S.at<cv::Vec3f>(i, j)[1] = a.at<cv::Vec3f>(i, j)[1] * Rp.at<cv::Vec3f>(i, j)[1] + b.at<cv::Vec3f>(i, j)[1];
			S.at<cv::Vec3f>(i, j)[2] = a.at<cv::Vec3f>(i, j)[2] * Rp.at<cv::Vec3f>(i, j)[2] + b.at<cv::Vec3f>(i, j)[2];
		}
	}*/
	return S;
}

cv::Mat CalcRGBmax(cv::Mat i_RGB)
{
	/*std::vector<cv::Mat> planes(3);
	cv::split(i_RGB, planes);
	cv::Mat m = cv::Mat(cv::max(planes[2], cv::max(planes[1], planes[0])));
	m = min(1, m);
	planes[0] = m;
	planes[1] = m;
	planes[2] = m;
	cv::Mat re;
	cv::merge(planes, re);
	return re;*/
	cv::Mat n(cv::Size(i_RGB.size()), CV_32FC3, cv::Scalar(0, 0, 0));
	for (int i = 0; i < i_RGB.rows; i++) {
		for (int j = 0; j < i_RGB.cols; j++) {
			float b = i_RGB.at<cv::Vec3f>(i, j)[0];
			float g = i_RGB.at<cv::Vec3f>(i, j)[1];
			float r = i_RGB.at<cv::Vec3f>(i, j)[2];
			float m = std::max(std::max(b, g), r);
			m = std::min(m, (float)1);
			n.at<cv::Vec3f>(i, j) = { m, m, m };
		}
	}
	return n;
}


#endif