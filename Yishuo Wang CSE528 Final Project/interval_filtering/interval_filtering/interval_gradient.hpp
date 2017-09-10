#ifndef INTERVAL_GRADIENT_HPP
#define INTERVAL_GRADIENT_HPP

#include <opencv2\highgui\highgui.hpp>
#include <opencv2\core\core.hpp>
#include <opencv2\opencv.hpp>
#include <iostream>
#include <vector>
#include <stdlib.h>

#include "rescaled.hpp"
#include "rebuild.hpp"

cv::Mat changeshape(cv::Mat image);
cv::Mat get_Window(int sigma, int window_Size);
cv::Mat interval_Filter(cv::Mat image, int sigma) {
	float epsi = pow(0.03, 2);
	int Nd = 3;
	int N = 8;

	cv::Mat wx_prev(cv::Size(image.size()), CV_32FC3, cv::Scalar(0, 0, 0));
	cv::Mat wy_prev(cv::Size(image.rows, image.cols), CV_32FC3, cv::Scalar(0, 0, 0));//change already
	cv::Mat wx;
	cv::Mat wy;
	
	cv::Mat S = image.clone();
	int times = 0;
	//get the window
	cv::Mat window = get_Window(sigma, ceil(3 * sigma));//19 * 1

	for (int ii = 0; ii < N; ii++) {
		cv::Mat rescaled_X = rescale(S, sigma, window);
		wx = get_w();

		cv::Mat rescaled_Y = rescale(changeshape(S), sigma, window);
		wy = get_w();

		std::cout << "filter " << times << "times" << std::endl;

		/////check stop//////
		float mean_dwx = 0, mean_dwy = 0;
		cv::Mat dwx = wx - wx_prev;
		cv::pow(dwx, 2, dwx);
		cv::Scalar dx = cv::sum(dwx);
		mean_dwx = (dx[0] + dx[1] + dx[2]) / (wx.rows * wx.cols * 3);

		cv::Mat dwy = wy - wy_prev;
		cv::pow(dwy, 2, dwy);
		cv::Scalar dy = cv::sum(dwy);
		mean_dwy = (dy[0] + dy[1] + dy[2]) / (wy.rows * wy.cols * 3);
		//for (int i = 0; i < wx.rows; i++) {//row
		//	for (int j = 0; j < wx.cols; j++) {//col
		//		mean_dwx += pow(wx.at<cv::Vec3f>(i, j)[0] - wx_prev.at<cv::Vec3f>(i, j)[0], 2);
		//		mean_dwx += pow(wx.at<cv::Vec3f>(i, j)[1] - wx_prev.at<cv::Vec3f>(i, j)[1], 2);
		//		mean_dwx += pow(wx.at<cv::Vec3f>(i, j)[2] - wx_prev.at<cv::Vec3f>(i, j)[2], 2);
		//	}
		//}
		//mean_dwx = mean_dwx / (wx.rows * wx.cols * 3);
		//for (int i = 0; i < wy.rows; i++) {//col
		//	for (int j = 0; j < wy.cols; j++) {//row
		//		mean_dwy += pow(wy.at<cv::Vec3f>(i, j)[0] - wy_prev.at<cv::Vec3f>(i, j)[0], 2);
		//		mean_dwy += pow(wy.at<cv::Vec3f>(i, j)[1] - wy_prev.at<cv::Vec3f>(i, j)[1], 2);
		//		mean_dwy += pow(wy.at<cv::Vec3f>(i, j)[2] - wy_prev.at<cv::Vec3f>(i, j)[2], 2);
		//	}
		//}
		//mean_dwy = mean_dwy / (wy.rows * wy.cols * 3);
		
		float diff = std::min(mean_dwx, mean_dwy);
		if (diff  <= 0.0025) {//paper use 0.0025
			break;
		}
		/////finish check stop//////

		S = image.clone();
		for (int i = 0; i < Nd; i++) {
			float sigmaI = sigma * 3 * sqrt(3) * pow(2, Nd - (i + 1)) / sqrt(pow(4, Nd) - 1);
			sigmaI = round(sigmaI);
			if (sigmaI == 0)
				break;
			//x
			S = reconstruct_1d(S, rescaled_X, sigmaI, epsi);
			S = changeshape(S);
			S = max(0, min(S, 1));
			//y
			S = reconstruct_1d(S, rescaled_Y, sigmaI, epsi);
			S = changeshape(S);
			S = max(0, min(S, 1));
		}
		wx_prev = wx;
		wy_prev = wy;
		cv::namedWindow("Filter Result", CV_WINDOW_FREERATIO);
		imshow("Filter Result", S);
		times++;
	}

	std::cout << times << std::endl;
	return S;
}
cv::Mat changeshape(cv::Mat image) {
	cv::Mat result(cv::Size(image.rows, image.cols), CV_32FC3, cv::Scalar(0, 0, 0));
	int row = 0, col = 0;
	for (int i = 0; i < result.rows; i++) {
		for (int j = 0; j < result.cols; j++) {
			result.at<cv::Vec3f>(i, j)[0] = image.at<cv::Vec3f>(row, col)[0];
			result.at<cv::Vec3f>(i, j)[1] = image.at<cv::Vec3f>(row, col)[1];
			result.at<cv::Vec3f>(i, j)[2] = image.at<cv::Vec3f>(row, col)[2];
			row++;
			if (row == image.rows) {
				col++;
				row = 0;
			}
		}
	}
	return result;
}
cv::Mat get_Window(int sigma, int window_Size) {
	cv::Mat win(cv::Size(2 * window_Size + 1, 1), CV_32FC1, cv::Scalar::all(0));
	float* helpwindow = new float[window_Size];
	float* window = new float[2 * window_Size + 1];
	int index = 0;
	float sum = 0;
	for (int i = window_Size - 1; i >= 0; i--) {//-8~0
		helpwindow[index++] = i * -1;
	}
	for (int i = 0; i < window_Size; i++) {
		helpwindow[i] = exp(-0.5 * pow(helpwindow[i] / sigma, 2));
		sum += helpwindow[i];
	}
	for (int i = 0; i < window_Size; i++) {
		helpwindow[i] = helpwindow[i] / sum;
	}
	window[0] = 0;
	index = 1;
	for (int i = 0; i < window_Size; i++) {
		window[index++] = helpwindow[i] * -1;
	}
	for (int i = window_Size - 1; i >= 0; i--) {
		window[index++] = helpwindow[i];
	}
	//build window kernel
	for (int i = 0; i < 2 * window_Size + 1; i++) {
		win.at<float>(0, i) = window[i];
	}
	//std::cout << win;
	return win;
}
#endif