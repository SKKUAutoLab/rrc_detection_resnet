#include "MainFunction.h"
#include "SupportFunction.h"

using namespace SupportFunction;
using namespace cv;

namespace MainFunction {
	/**
	* This function produce the simple road segmentation based on 2 input grayscale image
	* \param blackImg: the black channel image
	* \param whiteImg: the white channel image
	* \param outImg: the simple segmentation result
	* \param kernel: windows sliding size of the black and white channel image
	*/
	void getSimpleSegmentation(Mat &blackImg, Mat &whiteImg, Mat &outImg, int kernel) {
		int width = blackImg.cols;
		int height = blackImg.rows;

		int colStep = width / kernel;
		int rowStep = height / kernel;

		int colExtra = width % kernel;
		int rowExtra = height % kernel;

		Mat crrArea;
		Mat diffImg = whiteImg - blackImg;
		outImg = Mat::zeros(blackImg.size(), CV_8UC3);

		int crrX = 0;
		int crrY = 0;
		Scalar crrV;
		int blackValue, whiteValue, diffValue;

		for (int i = 0; i < colStep; ++i) {
			crrY = 0;
			for (int j = 0; j < rowStep; ++j) {
				blackValue = blackImg.at<uchar>(crrY, crrX);
				whiteValue = whiteImg.at<uchar>(crrY, crrX);
				diffValue = diffImg.at<uchar>(crrY, crrX);

				crrV = getSimpleSegmentationValue(blackValue, whiteValue, diffValue);

				crrArea = outImg(Rect(crrX, crrY, kernel, kernel));
				crrArea.setTo(crrV);

				crrY += kernel;
			}

			if (rowExtra > 0) {
				blackValue = blackImg.at<uchar>(crrY, crrX);
				whiteValue = whiteImg.at<uchar>(crrY, crrX);
				diffValue = diffImg.at<uchar>(crrY, crrX);

				crrV = getSimpleSegmentationValue(blackValue, whiteValue, diffValue);

				crrArea = outImg(Rect(crrX, crrY, kernel, rowExtra));
				crrArea.setTo(crrV);
			}

			crrX += kernel;
		}

		if (colExtra > 0) {
			crrY = 0;
			for (int i = 0; i < rowStep; ++i) {
				blackValue = blackImg.at<uchar>(crrY, crrX);
				whiteValue = whiteImg.at<uchar>(crrY, crrX);
				diffValue = diffImg.at<uchar>(crrY, crrX);

				crrV = getSimpleSegmentationValue(blackValue, whiteValue, diffValue);

				crrArea = outImg(Rect(crrX, crrY, colExtra, kernel));
				crrArea.setTo(crrV);

				crrY += kernel;
			}

			if (rowExtra > 0) {
				blackValue = blackImg.at<uchar>(crrY, crrX);
				whiteValue = whiteImg.at<uchar>(crrY, crrX);
				diffValue = diffImg.at<uchar>(crrY, crrX);
				
				crrV = getSimpleSegmentationValue(blackValue, whiteValue, diffValue);

				crrArea = outImg(Rect(crrX, crrY, colExtra, rowExtra));
				crrArea.setTo(crrV);
			}
		}
	}

	//-------------------------------------------------------------------------------------------------//
	/**
	* This function produce the simple road segmentation value based on 3 input grayscale value
	* \param blackV: the grayscale value from the black channel image
	* \param whiteV: the grayscale value from the the white channel image
	* \param diffV: the grayscale value from the the different channel image
	* Return the scalar value which represent the label of the checking area
	*/
	Scalar getSimpleSegmentationValue(int blackV, int whiteV, int diffV) {
		Scalar result;
		int blackTop, blackBot, whiteTop, whiteBot, diffTop, diffBot;
		blackTop = 100;
		blackBot = 20;
		whiteTop = 160;
		whiteBot = 50;
		diffTop = 60;
		diffBot = 20;

		if (blackV > blackBot && blackV < blackTop && whiteV < whiteTop 
			&& diffV < diffTop && diffV > diffBot) result = Scalar(0, 0, 255);

		return result;
	}
}