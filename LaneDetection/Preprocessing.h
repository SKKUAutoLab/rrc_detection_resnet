#include <opencv2/opencv.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/core/utility.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace cv;

namespace Preprocessing {
	/**
	* This function convert RGB inpImg to black channel outImg with a filter size kernel
	* \param inpImg: input RGB image
	* \param outImg: black channel image
	* \param kernel: windows sliding size
	*/
	void getBlackChannel(Mat &inpImg, Mat &outImg, int kernel = 15);

	//-------------------------------------------------------------------------------------------------//
	/**
	* This function return the minimum value in the 3 channel from the input RGB area
	* \param area: input RGB area
	*/
	Scalar getMinColorValue(Mat &area);

	//-------------------------------------------------------------------------------------------------//
	/**
	* This function convert RGB inpImg to white channel outImg with a filter size kernel
	* \param inpImg: input RGB image
	* \param outImg: white channel image (get the highest value from the RGB input region)
	* \param kernel: windows sliding size
	*/
	void getWhiteChannel(Mat &inpImg, Mat &outImg, int kernel = 15);

	//-------------------------------------------------------------------------------------------------//
	/**
	* This function return the maximum value in the 3 channel from the input RGB area
	* \param area: input RGB area
	*/
	uchar getMaxColorValue(Mat &area);
};
