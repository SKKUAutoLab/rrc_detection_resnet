#include "SupportFunction.h"

namespace SupportFunction {
	//-------------------------------------------------------------------------------------------------//
	/**
	* This function get the suitable IPM Matrix for the input image
	* \param inpImg: 3 channels color image
	* \param ipm: the ipm object which contains the transform matrix
	*/
	void getIPMMat(Mat &inpImg, IPM &ipm)
	{
		RoadType crrType;

		crrType = getRoadType(inpImg);

		if (ipm.roadType != crrType)
		{
			ipm = IPM();
			ipm.roadType = crrType;
			getMatrixByType(ipm, inpImg.size());
		}

	}

	//-------------------------------------------------------------------------------------------------//
	/**
	* This function get the roadtype for IPM function from the input image
	* \param inpImg: 3 channels color image
	* Return: 1 of the 3 road types declared in enum RoadType
	*/
	RoadType getRoadType(Mat &inpImg) 
	{
		RoadType rs = Mid;

		return rs;
	}

	//-------------------------------------------------------------------------------------------------//
	/**
	* This function return the mapping matrix for ipm object
	* \param ipm: IPM object contains all information for IPM function
	* \param size: size of the input image
	*/
	void getMatrixByType(IPM &ipm, Size size)
	{
		vector<Point2f> origPoints, dstPoints;
		RoadType type = (RoadType)ipm.roadType;

		int width = size.width;
		int height = size.height;

		//int scaleRate = width / height * 3;
		int scaleRate = 10;

		float heightRateTopOrig = 1. / 2;
		float heightRateBotOrig = 1.;
		float widthRateOrigTop = (scaleRate - 1) / 2. / scaleRate;
		float widthRateOrigBot = 1. / 16;

		cout << scaleRate << " " << widthRateOrigTop << endl;

		float widthRateDst = 3. / 8;
		float heightRateTopDst = 1. / 4;
		float heightRateBotDst = 1.;
		
		int origBot = height * heightRateBotOrig;
		int origTop = height * heightRateTopOrig;
		int origLeftBot = width * widthRateOrigBot;
		int origRightBot = width - origLeftBot;
		int origLeftTop = width * widthRateOrigTop;
		int origRightTop = width - origLeftTop;


		//cout << origBot << " " << origTop << " " << origLeft << " " << origRight << endl;

		Size dstSize = Size(400, 400*2);
		int dstLeft = dstSize.width * widthRateDst;
		int dstRight = dstSize.width - dstLeft;
		int dstTop = dstSize.height * heightRateTopDst;
		int dstBot = dstSize.height * heightRateBotDst;

		switch (type)
		{
		case Left:
			break;

		case Mid:
			/*
			origPoints.push_back(Point2f(width >> 3, height));
			origPoints.push_back(Point2f(width - (width >> 3), height));

			//origPoints.push_back(Point2f(width / 2 + 90, 386));  // vanishing point of Hyundai Dataset = 386
			//origPoints.push_back(Point2f(width / 2 - 90, 386));   // vanishing point of Hyundai Dataset = 386

			//origPoints.push_back(Point2f(width / 2 + 120, 386));  // vanishing point of Hyundai Dataset = 386
			//origPoints.push_back(Point2f(width / 2 - 105, 386));   // vanishing point of Hyundai Dataset = 386

			origPoints.push_back(Point2f(width / 2 + 140, 386));  // vanishing point of Hyundai Dataset = 386
			origPoints.push_back(Point2f(width / 2 - 120, 386));   // vanishing point of Hyundai Dataset = 386

			//origPoints.push_back(Point2f(width / 2 + 250, 386));  // vanishing point of Hyundai Dataset = 386
			//origPoints.push_back(Point2f(width / 2 + 50, 386));   // vanishing point of Hyundai Dataset = 386

			// The 4-points correspondences in the destination image			
			dstPoints.push_back(Point2f(width / 2 - 50, height));
			dstPoints.push_back(Point2f(width / 2 + 30, height));
			dstPoints.push_back(Point2f(width / 2 + 30, height / 2));
			dstPoints.push_back(Point2f(width / 2 - 50, height / 2));
			*/

			origPoints.push_back(Point2f(origLeftBot, origBot));
			origPoints.push_back(Point2f(origRightBot, origBot));
			origPoints.push_back(Point2f(origRightTop, origTop));
			origPoints.push_back(Point2f(origLeftTop, origTop));
			

			dstPoints.push_back(Point2f(dstLeft, dstBot));
			dstPoints.push_back(Point2f(dstRight, dstBot));
			dstPoints.push_back(Point2f(dstRight, dstTop));
			dstPoints.push_back(Point2f(dstLeft, dstTop));
			

			break;

		case Right:
			break;

		default:
			break;
		}

		ipm = IPM(size, dstSize, origPoints, dstPoints);
		return;
	}

	//-------------------------------------------------------------------------------------------------//
	/**
	* This function return the coordinates (X, Y) of 4 corners of the input BEV image
	* \param img: input grayscale BEV image
	* \param points: vector of 4 corner-points
	*/
	void getCornerOfBEV(Mat &img, vector<Point2d> &points) {
		int width = img.cols;
		int height = img.rows;
		
		points.clear();

		Point2d topLeft, topRight, botLeft, botRight;
		topLeft.y = topRight.y = 0;
		botLeft.y = botRight.y = height - 1;

		uchar crrV;
		int check1, check2;
		check1 = check2 = 0;

		for (int x = 0; x < width; ++x) {			
			crrV = img.at<uchar>(0, x);
			if (crrV > 0) {
				if (check1 == 0) {
					check1++;
					topLeft.x = x - 1;
				}
			}
			else {
				if (check1 == 1) {
					topRight.x = x;
					check1++;
				}
			}

			crrV = img.at<uchar>(height - 1, x);
			if (crrV > 0) {
				if (check2 == 0) {
					check2++;
					botLeft.x = x - 1;
				}
			}
			else {
				if (check2 == 1) {
					botRight.x = x;
					check2++;
				}
			}			
		}

		points.push_back(topLeft);
		points.push_back(topRight);
		points.push_back(botLeft);
		points.push_back(botRight);
	}

	//-------------------------------------------------------------------------------------------------//
	/**
	* This function print out the points in the input vector
	* \param points: vector of 2d points
	*/
	void printPoints(vector<Point2d> points) {
		for (int i = 0; i < points.size(); ++i) {
			if (i < points.size() - 1)
				cout << points[i] << " -- ";
			else
				cout << points[i];
		}
		cout << endl;
	}

	//-------------------------------------------------------------------------------------------------//
	/**
	* This function convert the BGR groungtruth image to binary image
	* \param inpImg: 3 channels color image
	*/
	void convertGTImg(Mat &inpImg) {
		Mat temp = inpImg.clone();
		inpImg = Mat::zeros(inpImg.size(), CV_8UC1);
		Vec3b crrV;

		for (int i = 0; i < temp.rows; ++i) {
			for (int j = 0; j < temp.cols; ++j) {
				crrV = temp.at<Vec3b>(i, j);
				if ((int)crrV[0] == 255 && (int)crrV[2] == 255) {
					inpImg.at<uchar>(i, j) = 255;
				}
			}
		}
		
	}
}