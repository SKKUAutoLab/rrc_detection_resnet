#include "Preprocessing.h"

namespace Preprocessing {
	/**
	* This function convert RGB inpImg to black channel outImg with a filter size kernel
	* \param inpImg: input RGB image
	* \param outImg: black channel image
	* \param kernel: windows sliding size
	*/
	void getBlackChannel(Mat &inpImg, Mat &outImg, int kernel) {
		int width = inpImg.cols;
		int height = inpImg.rows;

		int colStep = width / kernel;
		int rowStep = height / kernel;

		int colExtra = width % kernel;
		int rowExtra = height % kernel;

		Mat crrArea;
		outImg = Mat::zeros(inpImg.size(), CV_8UC1);

		int crrX = 0;
		int crrY = 0;
		Scalar crrV;

		for (int i = 0; i < colStep; ++i) {
			crrY = 0;
			for (int j = 0; j < rowStep; ++j) {				
				crrArea = inpImg(Rect(crrX, crrY, kernel, kernel));
				crrV = getMinColorValue(crrArea);
				crrArea = outImg(Rect(crrX, crrY, kernel, kernel));
				crrArea.setTo(crrV);
							
				crrY += kernel;
			}

			if (rowExtra > 0) {
				crrArea = inpImg(Rect(crrX, crrY, kernel, rowExtra));
				crrV = getMinColorValue(crrArea);
				crrArea = outImg(Rect(crrX, crrY, kernel, rowExtra));
				crrArea.setTo(crrV);
			}

			crrX += kernel;
		}

		if (colExtra > 0) {
			crrY = 0;
			for (int i = 0; i < rowStep; ++i) {
				crrArea = inpImg(Rect(crrX, crrY, colExtra, kernel));
				crrV = getMinColorValue(crrArea);
				crrArea = outImg(Rect(crrX, crrY, colExtra, kernel));
				crrArea.setTo(crrV);

				crrY += kernel;
			}		

			if (rowExtra > 0) {
				crrArea = inpImg(Rect(crrX, crrY, colExtra, rowExtra));
				crrV = getMinColorValue(crrArea);
				crrArea = outImg(Rect(crrX, crrY, colExtra, rowExtra));
				crrArea.setTo(crrV);
			}
		}
	}

	//-------------------------------------------------------------------------------------------------//
	/**
	* This function return the minimum value in the 3 channel from the input RGB area
	* \param area: input RGB area
	*/
	Scalar getMinColorValue(Mat &area) {
		int width = area.cols;
		int height = area.rows;
		int result = 255;

		for (int x = 0; x < width; ++x) {
			for (int y = 0; y < height; ++y) {
				Vec3b crrV = area.at<Vec3b>(y, x);
				for (int i = 0; i < 3; ++i) {
					if (crrV[i] < result) result = crrV[i];
				}
			}
		}

		return Scalar(result);
	}

	//-------------------------------------------------------------------------------------------------//
	/**
	* This function convert RGB inpImg to white channel outImg with a filter size kernel
	* \param inpImg: input RGB image
	* \param outImg: white channel image (get the highest value from the RGB input region)
	* \param kernel: windows sliding size
	*/
	void getWhiteChannel(Mat &inpImg, Mat &outImg, int kernel) {
		int width = inpImg.cols;
		int height = inpImg.rows;

		int colStep = width / kernel;
		int rowStep = height / kernel;

		int colExtra = width % kernel;
		int rowExtra = height % kernel;

		Mat crrArea;
		outImg = Mat::zeros(inpImg.size(), CV_8UC1);

		int crrX = 0;
		int crrY = 0;
		int crrV;

		for (int i = 0; i < colStep; ++i) {
			crrY = 0;
			for (int j = 0; j < rowStep; ++j) {
				crrArea = inpImg(Rect(crrX, crrY, kernel, kernel));
				crrV = getMaxColorValue(crrArea);
				crrArea = outImg(Rect(crrX, crrY, kernel, kernel));
				crrArea.setTo(Scalar(crrV));

				crrY += kernel;
			}

			if (rowExtra > 0) {
				crrArea = inpImg(Rect(crrX, crrY, kernel, rowExtra));
				crrV = getMaxColorValue(crrArea);
				crrArea = outImg(Rect(crrX, crrY, kernel, rowExtra));
				crrArea.setTo(Scalar(crrV));
			}

			crrX += kernel;
		}

		if (colExtra > 0) {
			crrY = 0;
			for (int i = 0; i < rowStep; ++i) {
				crrArea = inpImg(Rect(crrX, crrY, colExtra, kernel));
				crrV = getMaxColorValue(crrArea);
				crrArea = outImg(Rect(crrX, crrY, colExtra, kernel));
				crrArea.setTo(Scalar(crrV));

				crrY += kernel;
			}

			if (rowExtra > 0) {
				crrArea = inpImg(Rect(crrX, crrY, colExtra, rowExtra));
				crrV = getMaxColorValue(crrArea);
				crrArea = outImg(Rect(crrX, crrY, colExtra, rowExtra));
				crrArea.setTo(Scalar(crrV));
			}
		}
	}

	//-------------------------------------------------------------------------------------------------//
	/**
	* This function return the minimum value in the 3 channel from the input RGB area
	* \param area: input RGB area
	*/
	uchar getMaxColorValue(Mat &area) {
		int width = area.cols;
		int height = area.rows;
		uchar result = 0;

		for (int x = 0; x < width; ++x) {
			for (int y = 0; y < height; ++y) {
				Vec3b crrV = area.at<Vec3b>(y, x);
				for (int i = 0; i < 3; ++i) {
					if (crrV[i] > result) result = crrV[i];
				}
			}
		}

		return result;
	}
}