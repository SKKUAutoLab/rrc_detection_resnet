#include "IPM.h"

using namespace std;
using namespace cv;

typedef enum roadType { Left = 0, Mid, Right } RoadType;

namespace SupportFunction {
	//-------------------------------------------------------------------------------------------------//
	/**
	* This function get the suitable IPM Matrix for the input image
	* \param inpImg: 3 channels color image
	* \param ipm: the ipm object which contains the transform matrix
	*/
	void getIPMMat(Mat &inpImg, IPM &ipm);

	//-------------------------------------------------------------------------------------------------//
	/**
	* This function get the roadtype for IPM function from the input image
	* \param inpImg: 3 channels color image
	* Return: 1 of the 3 road types declared in enum RoadType
	*/
	RoadType getRoadType(Mat &inpImg);

	//-------------------------------------------------------------------------------------------------//
	/**
	* This function return the mapping matrix for ipm object
	* \param ipm: IPM object contains all information for IPM function
	* \param size: size of the input image
	*/
	void getMatrixByType(IPM &ipm, Size size);

	//-------------------------------------------------------------------------------------------------//
	/**
	* This function return the coordinates (X, Y) of 4 corners of the input BEV image
	* \param img: input grayscale BEV image
	* \param points: vector of 4 corner-points
	*/
	void getCornerOfBEV(Mat &img, vector<Point2d> &points);

	//-------------------------------------------------------------------------------------------------//
	/**
	* This function print out the points in the input vector
	* \param points: vector of 2d points
	*/
	void printPoints(vector<Point2d> points);

	//-------------------------------------------------------------------------------------------------//
	/**
	* This function convert the BGR groungtruth image to binary image
	* \param inpImg: 3 channels color image
	*/
	void convertGTImg(Mat &inpImg);
};