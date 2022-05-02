#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <vector>

#include "IPM.h"
#include "GetSetting.h"

using namespace cv;
using namespace std;

typedef struct HistRegion {
	int startX;
	int endX;
	int midX;
	int peakValue;
	int totalValue;

	HistRegion(int start, int end, int mid, int peak, int total) {
		startX = start;
		endX = end;
		midX = mid;
		peakValue = peak;
		totalValue = total;
	}

	HistRegion() {
		startX = -1;
		endX = -1;
		midX = -1;
		peakValue = -1;
		totalValue = -1;
	}

} HistRegion;

namespace EstimateLaneRegion {
	//-------------------------------------------------------------------------------------------------//
	/**
	* This function produce an estimated lane area image from the RGB input image
	*/
	void getEstimateLaneRegion(Mat &inpImg, Mat &outImg, GetSetting laneRegionSetting);
	
	//-------------------------------------------------------------------------------------------------//
	/**
	* This function draw an estimated lane region
	* from set of cornerPoints and fill with regionColor value
	* The result is RGB color image contains only 2 color, black and regionColor
	*/
	void drawRegionFromBEV(Mat &inpImg, Mat &outImg, vector<Point> &cornerPoints, GetSetting laneRegionSetting);

	//-------------------------------------------------------------------------------------------------//
	/**
	* This function get edge image based on the algorithm in my paper
	*/
	void edgeDetection(Mat &inpImg, Mat &edgeImg, GetSetting laneRegionSetting);

	//-------------------------------------------------------------------------------------------------//
	/**
	* This function get 3 threshold value for custom edge detection algorithm
	* by using histogram of the input image (grayscale)
	*/
	void getThresholdValueFromImage(Mat &inpImg, int &thresh1, int &thresh2, int &thresh3, GetSetting laneRegionSetting);

	//-------------------------------------------------------------------------------------------------//
	/**
	* This function get lane type from edge image by using
	* custom histogram algorithm
	*/
	void getLanePointsFromEdgeImage(Mat &inpImg, Mat &edgeImg, int &originLeftX, int &originRightX, vector<Point> &cornerPoints, GetSetting laneRegionSetting);

	//-------------------------------------------------------------------------------------------------//
	/**
	* This function get the seed left and right x-coordinate from
	* histogram of the edge image
	*/
	void findSeedPointFromHist(Mat &histImg, int &seedLeftX, int &seedRightX, GetSetting laneRegionSetting);

	//-------------------------------------------------------------------------------------------------//
	/**
	* This function get the left and right x-coordinate from
	* input histogram image and update the mid position also
	*/
	void getLanePositionFromHistogram(Mat histImg, int &crrLeft, int &crrRight, int &crrMid, int maxHistValue, int histPosition, int &momentumLeft, int &momentumRight, GetSetting laneRegionSetting);

	//-------------------------------------------------------------------------------------------------//
	/**
	* This function merge too closed hist region in the set of
	& calculated input hist regions
	*/
	void mergeHistRegion(vector<HistRegion> &histRegion, int maxHist);

	//-------------------------------------------------------------------------------------------------//
	/**
	* This function draw an image based on the input histogram
	* and input width and height
	*/
	void drawHistToImage(Mat inpHist, Mat &histImg, int width, int height, int maxHist, int scale = 1);

	//-------------------------------------------------------------------------------------------------//
	/**
	* This function return a histogram of amount of pixel within each column
	* from the ROI of the input image
	*/
	void getColumnHistFromROI(Mat &inpImg, int botRow, int topRow, Mat &histImg, int binSize = 1);

	//-------------------------------------------------------------------------------------------------//
	/**
	* This function get the suitable IPM Matrix for the input image
	* \param inpImg: 3 channels color image
	* \param ipm: the ipm object which contains the transform matrix
	*/
	void getIPM(Mat &inpImg, IPM &ipm, GetSetting laneRegionSetting);


}
