#include <opencv2/opencv.hpp>
//#include <opencv2/ximgproc.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/core/utility.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>


using namespace cv;

namespace MainFunction {
	/**
	* This function produce the simple road segmentation based on 2 input grayscale image
	* \param blackImg: the black channel image
	* \param whiteImg: the white channel image
	* \param outImg: the simple segmentation result
	* \param kernel: windows sliding size of the black and white channel image
	*/
	void getSimpleSegmentation(Mat &blackImg, Mat &whiteImg, Mat &outImg, int kernel);
	
	//-------------------------------------------------------------------------------------------------//
	/**
	* This function produce the simple road segmentation value based on 3 input grayscale value
	* \param blackV: the grayscale value from the black channel image
	* \param whiteV: the grayscale value from the the white channel image
	* \param diffV: the grayscale value from the the different channel image
	* Return the scalar value which represent the label of the checking area
	*/
	Scalar getSimpleSegmentationValue(int blackV, int whiteV, int diffV);
};
