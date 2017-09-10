#ifndef RESCALED_HPP
#define RESCALED_HPP

#include <opencv2\highgui\highgui.hpp>
#include <opencv2\core\core.hpp>
#include <opencv2\opencv.hpp>
#include <iostream>
#include <vector>
#include <stdlib.h>


cv::Mat get_nor_grad(cv::Mat image);
cv::Mat get_Inte_Grad(int sigma, int window_Size, cv::Mat image, cv::Mat window);
cv::Mat get_Inte(int window_Size, int x, int y, cv::Mat image, cv::Mat window);
cv::Mat getMaxBgrArea(cv::Mat bgr);
cv::Mat getCorElement(cv::Mat orgi, cv::Mat match);
cv::Mat getCorNum(cv::Mat nor_grad);
//rescaled gradient, wx

cv::Mat public_w;

cv::Mat rescale(cv::Mat image, int sigma, cv::Mat window) {
	int window_Size = ceil(3 * sigma);
	float err = pow(10, -4);
	//get normal gradient
	cv::Mat nor_grad = get_nor_grad(image);//3 channel
	
	//get the interval gradient
	cv::Mat Inte_Grad = get_Inte_Grad(sigma, window_Size, image, window);//3 channel

	cv::Mat res_Grad(cv::Size(image.size()), CV_32FC3, cv::Scalar(0, 0, 0));//3 channel

	cv::Mat corNum = getCorNum(nor_grad);
	cv::Mat top = getCorElement(Inte_Grad, corNum);
	cv::Mat bot = getCorElement(nor_grad, corNum);

	//for (int i = 1; i < top.rows; i++) {//row
	//	for (int j = 0; j < top.cols; j++) {//col
	//		std::cout << top.at<float>(i, j) << std::endl;
	//	}
	//}
	//std::cout << "RG" << std::endl;
	
	cv::Mat wx = (abs(top) + err) / (abs(bot) + err);//1 channel
	wx = min(wx, 1);

	//std::cout << "RG" << std::endl;
	//for (int i = 0; i < wx.rows; i++) {//row
	//	for (int j = 0; j < wx.cols; j++) {//col
	//		std::cout << wx.at<float>(i, j) << std::endl;
	//	}
	//}
	//std::cout << "RG" << std::endl;

	for (int i = 0; i < res_Grad.rows; i++) {
		for (int j = 0; j < res_Grad.cols; j++) {
			if (top.at<float>(i, j) * bot.at<float>(i, j) <= 0){
				res_Grad.at<cv::Vec3f>(i, j) = { 0, 0, 0 };
			}
			else{
				res_Grad.at<cv::Vec3f>(i, j) = nor_grad.at<cv::Vec3f>(i, j) * wx.at<float>(i, j);
			}
		}
	}
	cv::Mat w(cv::Size(image.size()), CV_32FC3, cv::Scalar(1, 1, 1));
	for (int i = 0; i < w.rows; i++) {
		for (int j = 0; j < w.cols; j++) {
			w.at<cv::Vec3f>(i, j) = w.at<cv::Vec3f>(i, j) * wx.at<float>(i, j);
		}
	}
	public_w = w;
	return res_Grad;
}
cv::Mat getMaxBgrArea(cv::Mat bgr){
	cv::Mat maxBgr(cv::Size(bgr.size()), CV_32FC1, cv::Scalar::all(0));
	for (int i = 0; i < maxBgr.rows; i++) {
		for (int j = 0; j < maxBgr.cols; j++) {
			float b = abs(bgr.at<cv::Vec3f>(i, j)[0]);
			float g = abs(bgr.at<cv::Vec3f>(i, j)[1]);
			float r = abs(bgr.at<cv::Vec3f>(i, j)[2]);
			if (r >= g && r >= b) 
				maxBgr.at<float>(i, j) = 0;
			else if (g >= r && g >= b)
				maxBgr.at<float>(i, j) = 1;
			else
				maxBgr.at<float>(i, j) = 2;
		}
	}
	return maxBgr;
}
cv::Mat getCorNum(cv::Mat nor_grad) {
	cv::Mat maxBgr = getMaxBgrArea(abs(nor_grad));
	cv::Mat tempp(cv::Size(nor_grad.size()), CV_32FC1, cv::Scalar::all(0));
	for (int i = 0; i < tempp.cols; i++) {
		for (int j = 0; j < tempp.rows; j++) {
			tempp.at<float>(j, i) = i * tempp.rows + j + 1;
		}
	}
	maxBgr = maxBgr * nor_grad.rows * nor_grad.cols + tempp;
	return maxBgr;
}

cv::Mat getCorElement(cv::Mat orgi, cv::Mat match){
	cv::Mat mea(cv::Size(match.size()), CV_32FC1, cv::Scalar::all(0));
	for (int i = 0; i < mea.rows; i++) {
		for (int j = 0; j < mea.cols; j++) {
			int num = (int)match.at<float>(i, j);

			int left = num % (orgi.rows * orgi.cols);
			int ch = 2 - num / (orgi.rows * orgi.cols);
			if (left == 0){
				ch = ch + 1;
				left = orgi.rows * orgi.cols;
			}
			
			int r = left % orgi.rows;
			int c = left / orgi.rows;
			if (r == 0){
				r = orgi.rows - 1;
				c = c - 1;
			}
			else{
				r = r - 1;
			}
			mea.at<float>(i, j) = orgi.at<cv::Vec3f>(r, c)[ch];
		}
	}
	return mea;
}
cv::Mat get_w() {
	return public_w;
}

cv::Mat get_nor_grad(cv::Mat image) {
	//cv::Mat nor_grad;
	cv::Mat win(cv::Size(3, 1), CV_32FC1, cv::Scalar::all(0));
	win.at<float>(0, 0) = 0;
	win.at<float>(0, 1) = -1;
	win.at<float>(0, 2) = 1;

	/*cv::Point anchor = cv::Point(-1, -1);
	cv::filter2D(image, nor_grad, -1, win, anchor, 0, cv::BORDER_REPLICATE);*/
	cv::Mat nor_grad(cv::Size(image.size()), CV_32FC3, cv::Scalar(0, 0, 0));
	for (int i = 0; i < nor_grad.rows; i++) {	
		for (int j = 0; j < nor_grad.cols - 1; j++) {
			nor_grad.at<cv::Vec3f>(i, j) = image.at<cv::Vec3f>(i, j + 1) - image.at<cv::Vec3f>(i, j);
		}
		nor_grad.at<cv::Vec3f>(i, nor_grad.cols - 1) = { 0, 0, 0 };
	}
	return nor_grad;
}

cv::Mat get_Inte_Grad(int sigma, int window_Size, cv::Mat image, cv::Mat window) {
	//opencv filter
	/*cv::Point anchor = cv::Point(-1, -1);
	cv::Mat inte_Grad;
	cv::filter2D(image, inte_Grad, -1, window, anchor, 0, cv::BORDER_REPLICATE);*/

	//or loop
	cv::Mat inte_Grad(cv::Size(image.size()), CV_32FC3, cv::Scalar(0, 0, 0));
	for (int i = 0; i < inte_Grad.rows; i++) {
		for (int j = 0; j < inte_Grad.cols; j++){
			inte_Grad.at<cv::Vec3f>(i, j) = get_Inte(window_Size, i, j, image, window).at<cv::Vec3f>(0, 0);
		}
	}
	return inte_Grad;
}
cv::Mat get_Inte(int window_Size, int x, int y, cv::Mat image, cv::Mat window) {//window_Size = 3x3 = 9
	float B = 0, G = 0, R = 0;
	int start = window_Size * -1;
	for (int i = 0; i < 2 * window_Size + 1; i++) {
		if (y + start < 0) {
			B += window.at<float>(0, i) * image.at<cv::Vec3f>(x, 0)[0];
			G += window.at<float>(0, i) * image.at<cv::Vec3f>(x, 0)[1];
			R += window.at<float>(0, i) * image.at<cv::Vec3f>(x, 0)[2];
		}
		else if (y + start >= image.cols) {
			B += window.at<float>(0, i) * image.at<cv::Vec3f>(x, image.cols - 1)[0];
			G += window.at<float>(0, i) * image.at<cv::Vec3f>(x, image.cols - 1)[1];
			R += window.at<float>(0, i) * image.at<cv::Vec3f>(x, image.cols - 1)[2];
		}
		else {
			B += window.at<float>(0, i) * image.at<cv::Vec3f>(x, y + start)[0];
			G += window.at<float>(0, i) * image.at<cv::Vec3f>(x, y + start)[1];
			R += window.at<float>(0, i) * image.at<cv::Vec3f>(x, y + start)[2];
		}
		start++;
	}

	cv::Mat BGR(cv::Size(1, 1), CV_32FC3, cv::Scalar(B, G, R));
	//std::cout << BGR << std::endl;
	return BGR;
}

#endif
