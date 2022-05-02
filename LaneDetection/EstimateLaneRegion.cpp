#include "EstimateLaneRegion.h"


#pragma warning(disable : 4996)
//=======================================================================================================//

namespace EstimateLaneRegion {
	//-------------------------------------------------------------------------------------------------//
	/**
	* This function produce an estimated lane area image from the RGB input image
	*/
	void getEstimateLaneRegion(Mat &inpImg, Mat &outImg, GetSetting laneRegionSetting) {
		Mat blurImg, originBEVImg, labBEVImg, grayBEVImg, regionBEVImg;
		vector<Mat> splitChannels;
		vector<Point> cornerPoints;
		IPM ipm;
		
		// Apply denoising filter
		blur(inpImg, blurImg, Size(5, 5));

		// Convert to BEV image
		getIPM(blurImg, ipm, laneRegionSetting);
		ipm.applyHomography(inpImg, originBEVImg);
		//if (laneRegionSetting.debug) imshow("Original bev image", originBEVImg);

		// Convert to gray image by using only L channel from Lab image
		labBEVImg = Mat(originBEVImg.size(), CV_8UC3);
		cvtColor(originBEVImg, labBEVImg, COLOR_BGR2Lab);
		split(labBEVImg, splitChannels);
		grayBEVImg = splitChannels[0];
		//if (laneRegionSetting.debug) imshow("Gray bev image", grayBEVImg);

		drawRegionFromBEV(grayBEVImg, regionBEVImg, cornerPoints, laneRegionSetting);
		//if (laneRegionSetting.debug) imshow("Lane region of bev image", regionBEVImg + originBEVImg);

		// Map back lane region result from BEV to normal space
		ipm.applyHomographyInv(regionBEVImg, outImg);
		//if (laneRegionSetting.debug) imshow("Final Lane region", outImg + inpImg);
	}

	//-------------------------------------------------------------------------------------------------//
	/**
	* This function draw an estimated lane region of BEV image
	* from set of cornerPoints and fill with regionColor value
	* The result is RGB color image contains only 2 color, black and regionColor
	*/
	void drawRegionFromBEV(Mat &bevImg, Mat &outImg, vector<Point> &cornerPoints, GetSetting laneRegionSetting) {
		int originLeftX = 145, originRightX = 255;
		Mat edgeImg;
		Scalar regionColor = Scalar(laneRegionSetting.outColorB, laneRegionSetting.outColorG, laneRegionSetting.outColorR);

		edgeDetection(bevImg, edgeImg, laneRegionSetting);
		//if (laneRegionSetting.debug) imshow("Edge of ROI BEV image", edgeImg);
		getLanePointsFromEdgeImage(bevImg, edgeImg, originLeftX, originRightX, cornerPoints, laneRegionSetting);
		outImg = Mat(bevImg.rows, bevImg.cols, CV_8UC3, Scalar(0, 0, 0));

		for (int i = 1; i < cornerPoints.size(); ++i) {
			line(outImg, cornerPoints[i - 1], cornerPoints[i], regionColor, 3, LINE_AA);
		}

		Point centerPoint = (cornerPoints[0] + cornerPoints[cornerPoints.size() - 1]) / 2;
		floodFill(outImg, centerPoint, regionColor, 0, Scalar(0, 0, 0), regionColor);
	}

	//-------------------------------------------------------------------------------------------------//
	/**
	* This function get edge image of ROI bevImg (bottom part 400x400)
	* based on the algorithm in my paper
	*/
	void edgeDetection(Mat &inpImg, Mat &edgeImg, GetSetting laneRegionSetting) {
		int threshValue1, threshValue2, threshValue3;
		int roiWidth = 400, roiHeight = 800;
		int compareDistance = laneRegionSetting.edgeDetectionKernel;
		
		edgeImg = Mat::zeros(roiHeight, roiWidth, CV_8U);

		getThresholdValueFromImage(inpImg, threshValue1, threshValue2, threshValue3, laneRegionSetting);
		if (laneRegionSetting.debug) {
			cout << threshValue1 << " -- " << threshValue2 << " -- " << threshValue3 << endl;
		}
		
		Mat roiBEV = inpImg(Rect(0, inpImg.rows - roiHeight, roiWidth, roiHeight)).clone();
		Mat shiftLeft1, shiftRight1, shiftLeft2, shiftRight2, padImg;

		//if (laneRegionSetting.debug) imshow("Bird\'s Eye Image", roiBEV);

		copyMakeBorder(roiBEV, padImg, 0, 0, compareDistance, compareDistance, BORDER_REPLICATE);
		shiftRight1 = Mat(padImg, Rect(compareDistance + compareDistance - 1, 0, roiWidth, roiHeight)).clone();
		shiftLeft1 = Mat(padImg, Rect(1, 0, roiWidth, roiHeight)).clone();

		shiftRight2 = Mat(padImg, Rect(compareDistance + compareDistance, 0, roiWidth, roiHeight)).clone();
		shiftLeft2 = Mat(padImg, Rect(0, 0, roiWidth, roiHeight)).clone();

		Mat dstLeft1 = roiBEV - shiftLeft1;
		Mat dstRight1 = roiBEV - shiftRight1;

		Mat dstLeft2 = roiBEV - shiftLeft2;
		Mat dstRight2 = roiBEV - shiftRight2;

		int valueLeft1, valueRight1, valueLeft2, valueRight2, crrV;
		int possibleLanePixel;

		for (int x = compareDistance; x < roiWidth - compareDistance; ++x) {
			for (int y = compareDistance; y < roiHeight; ++y) {
				possibleLanePixel = 0;
				crrV = roiBEV.at<uchar>(y, x);

				for (int i = -2; i < 3; ++i) {
					for (int j = -2; j < 3; ++j) {
						if ((int)roiBEV.at<uchar>(y + j, x + i) > threshValue1)
							possibleLanePixel++;
					}
				}
				if (possibleLanePixel < 15) {
					edgeImg.at<uchar>(y, x) = 0;
					continue;
				}

				valueLeft1 = dstLeft1.at<uchar>(y, x);
				valueRight1 = dstRight1.at<uchar>(y, x);
				valueLeft2 = dstLeft2.at<uchar>(y, x);
				valueRight2 = dstRight2.at<uchar>(y, x);

				if (crrV > threshValue1 && valueLeft1 > -5 && valueRight1 > -5 && (valueLeft1 + valueRight1) > threshValue2)
					edgeImg.at<uchar>(y, x) = 255;

				if (edgeImg.at<uchar>(y, x) == 255) continue;

				if (crrV > threshValue1 && valueLeft2 > -5 && valueRight2 > -5 && (valueLeft2 + valueRight2) > threshValue3)
					edgeImg.at<uchar>(y, x) = 255;
			}
		}

		//if (laneRegionSetting.debug) imshow("Edge before morphology", edgeImg);	

		Mat verticalStructure = getStructuringElement(MORPH_RECT, Size(2, 2));
		erode(edgeImg, edgeImg, verticalStructure, Point(-1, -1));

		verticalStructure = getStructuringElement(MORPH_RECT, Size(3, 3));
		dilate(edgeImg, edgeImg, verticalStructure, Point(-1, -1));
	}

	//-------------------------------------------------------------------------------------------------//
	/**
	* This function get 3 threshold value for custom edge detection algorithm
	* by using histogram of the input image (grayscale)
	*/
	void getThresholdValueFromImage(Mat &inpImg, int &thresh1, int &thresh2, int &thresh3, GetSetting laneRegionSetting) {
		int width = inpImg.cols, height = inpImg.rows;
		int histSize = 256;
		float range[] = { 0, 256 };
		const float* histRange = { range };
		bool uniform = true; bool accumulate = false;
		Mat gray_hist;

		calcHist(&inpImg, 1, 0, Mat(), gray_hist, 1, &histSize, &histRange, uniform, accumulate);
		
		int hist_w = 512; int hist_h = 400;
		int bin_w = cvRound((double)hist_w / histSize);
		Mat histImg(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
		if (laneRegionSetting.debug) {
			normalize(gray_hist, gray_hist, 0, histImg.rows, NORM_MINMAX, -1, Mat());
			for (int i = 1; i < histSize; i++) {

				line(histImg, Point(bin_w*(i - 1), hist_h - cvRound(gray_hist.at<float>(i - 1))),
					Point(bin_w*(i), hist_h - cvRound(gray_hist.at<float>(i))),
					Scalar(255, 0, 0), 2, 8, 0);
			}
		}
		//if (laneRegionSetting.debug) imshow("Hist image of the whole BEV", histImg);

		double s = cv::sum(gray_hist)[0];
		if (laneRegionSetting.debug) cout << " --- total hist s = " << s << endl;
		s -= gray_hist.at<float>(0);
		s *= laneRegionSetting.meanHistogramFindingRate;

		if (laneRegionSetting.debug) cout << " --- s = " << s << endl;

		vector<pair<float, int>> sortedHist;
		int tempIndex = -1;

		sortedHist.push_back(make_pair(0, 0));
		for (int i = 1; i < histSize; i++)
		{
			sortedHist.push_back(make_pair(gray_hist.at<float>(i) + sortedHist[i - 1].first, i));
			if (sortedHist[i].first > s && tempIndex < 0) tempIndex = i;
		}

		thresh1 = tempIndex;
		thresh2 = min((float)thresh1 * 0.25, 15.0);
		thresh3 = min((float)thresh1 * 0.3, 25.0);		
	}

	//-------------------------------------------------------------------------------------------------//
	/**
	* This function get lane type from edge image by using
	* custom histogram algorithm
	*/
	void getLanePointsFromEdgeImage(Mat &inpImg, Mat &edgeImg, int &originLeftX, int &originRightX, vector<Point> &cornerPoints, GetSetting laneRegionSetting) {
		int width = inpImg.cols, height = inpImg.rows;
		int edgeWidth = edgeImg.cols, edgeHeight = edgeImg.rows;

		int histWidth, midHist;
		Mat tempHistRegion;
		vector<Mat> histRegion;
		vector<int> maxHistValue;
		int startHistRow, endHistRow;
		int numHist = laneRegionSetting.numOfHistForGettingLanePoints;
		int histStep = edgeHeight / numHist;
		int seedLeftX, seedRightX;

		startHistRow = endHistRow = edgeHeight;

		getColumnHistFromROI(edgeImg, 0, edgeHeight, tempHistRegion);
		histRegion.push_back(tempHistRegion);
		maxHistValue.push_back(edgeHeight - 0);

		//findSeedPointFromHist(histRegion[0], seedLeftX, seedRightX, laneRegionSetting);

		// ----- New Improvement here -----
		if (laneRegionSetting.inpWidth <= 0) {
			seedLeftX = 130;
			seedRightX = 270;
		}
		else if (laneRegionSetting.inpWidth < 1500) {
			seedLeftX = width / 40 * 13;
			seedRightX = width - seedLeftX;
		}
		else if (laneRegionSetting.inpWidth < 2400) {
			seedLeftX = width / 40 * 15;
			seedRightX = width - seedLeftX;
		}
		else {
			seedLeftX = width / 40 * 17;
			seedRightX = width - seedLeftX;
		}
		// ----------------------

		for (int i = 0; i < numHist; ++i) {
			startHistRow = max(0, startHistRow - histStep);
			getColumnHistFromROI(edgeImg, startHistRow, endHistRow, tempHistRegion);
			histRegion.push_back(tempHistRegion);
			maxHistValue.push_back(endHistRow - startHistRow);
			endHistRow -= histStep;
		}

		histWidth = histRegion[1].cols;
		midHist = histWidth >> 1;

		vector<Point> leftPoints, rightPoints;
		int crrMid, crrLeft, crrRight;
		Point leftPoint, rightPoint;
		int momentumLeft = 0, momentumRight = 0;

		crrMid = midHist;
		crrLeft = seedLeftX, crrRight = seedRightX;
		leftPoint.y = rightPoint.y = height - 1;

		if (laneRegionSetting.debug) {
			cout << "seed left = " << seedLeftX << " || seed right = " << seedRightX << endl;
		}		

		// ----- New Improvement here -----
		leftPoint.x = crrLeft;
		rightPoint.x = crrRight;

		leftPoints.push_back(leftPoint);
		rightPoints.push_back(rightPoint);
		
		leftPoint.y -= edgeHeight >> 2;
		rightPoint.y -= edgeHeight >> 2;
		// --------------------------------

		//for (int i = 1; i < histRegion.size(); ++i) {
		for (int i = 2; i < histRegion.size(); ++i) {
			getLanePositionFromHistogram(histRegion[i], crrLeft, crrRight, crrMid, maxHistValue[i], i, momentumLeft, momentumRight, laneRegionSetting);

			leftPoint.x = crrLeft;
			rightPoint.x = crrRight;

			if (i == 3) {
				leftPoint.y -= edgeHeight >> 2;
				rightPoint.y -= edgeHeight >> 2;
			}

			leftPoints.push_back(leftPoint);
			rightPoints.push_back(rightPoint);

			if (laneRegionSetting.debug) {
				cout << leftPoint << " === " << rightPoint << endl;
			}			

			leftPoint.y -= edgeHeight >> 2;
			rightPoint.y -= edgeHeight >> 2;
		}

		for (int i = 0; i < leftPoints.size(); ++i) {
			cornerPoints.push_back(leftPoints[i]);
		}

		for (int i = rightPoints.size() - 1; i >= 0; --i) {
			cornerPoints.push_back(rightPoints[i]);
		}
		

		//----- Debug Part to visualize the histogram images -----------------
		//Mat tempHistImg;
		//vector<Mat> histImg;
		//char windowName[100];

		//if (laneRegionSetting.debug) {
		//	for (int i = 0; i < histRegion.size(); ++i) {
		//		drawHistToImage(histRegion[i], tempHistImg, 400, 800, maxHistValue[i], 2);
		//		histImg.push_back(tempHistImg);
		//		sprintf(windowName, "Hist %d", i + 1);
		//		imshow(windowName, histImg[i]);
		//	}
		//}
	}

	//-------------------------------------------------------------------------------------------------//
	/**
	* This function get the seed left and right x-coordinate from
	* histogram of the edge image
	*/
	void findSeedPointFromHist(Mat &histImg, int &seedLeftX, int &seedRightX, GetSetting laneRegionSetting) {
		int checkRange = histImg.cols >> 2;
		int maxLeft, maxRight;
		int crrLeft, crrRight;
		int midX = histImg.cols >> 1;
		int minLen = laneRegionSetting.minLenToFindSeedPoint;
		int leftBoundary, rightBoundary;

		seedLeftX = seedRightX = midX;
		leftBoundary = midX - checkRange;
		rightBoundary = midX + checkRange;

		int oldSeedLeft, oldSeedRight;
		bool checkMore = false;

		while (seedRightX - seedLeftX < minLen) {
			maxLeft = maxRight = 0;
			crrLeft = seedLeftX;
			crrRight = seedRightX;

			oldSeedLeft = seedLeftX;
			oldSeedRight = seedRightX;

			if (crrRight - midX < midX - crrLeft) {
				crrRight++;
			}
			else {
				crrLeft--;
			}

			for (int i = 0; i < checkRange; ++i) {
				if (histImg.at<float>(crrLeft) > maxLeft && crrLeft > leftBoundary) {
					maxLeft = histImg.at<float>(crrLeft);
					seedLeftX = crrLeft;
				}

				if (histImg.at<float>(crrRight) > maxRight && crrRight < rightBoundary) {
					maxRight = histImg.at<float>(crrRight);
					seedRightX = crrRight;
				}

				crrLeft--;
				crrRight++;
			}

			if (oldSeedLeft == seedLeftX && oldSeedRight == seedRightX) {
				checkMore = true;
				break;
			}
		}

		if (!checkMore) return;

		while (seedRightX - seedLeftX < minLen) {
			maxLeft = maxRight = 0;
			crrLeft = seedLeftX;
			crrRight = seedRightX;

			if (crrRight - midX < midX - crrLeft) {
				crrRight++;
			}
			else {
				crrLeft--;
			}

			for (int i = 0; i < checkRange; ++i) {
				if (histImg.at<float>(crrLeft) >= maxLeft && crrLeft > leftBoundary) {
					maxLeft = histImg.at<float>(crrLeft);
					seedLeftX = crrLeft;
				}

				if (histImg.at<float>(crrRight) >= maxRight && crrRight < rightBoundary) {
					maxRight = histImg.at<float>(crrRight);
					seedRightX = crrRight;
				}

				crrLeft--;
				crrRight++;
			}
		}

	}

	//-------------------------------------------------------------------------------------------------//
	/**
	* This function get the left and right x-coordinate from
	* input histogram image and update the mid position also
	*/
	void getLanePositionFromHistogram(Mat histImg, int &crrLeft, int &crrRight, int &crrMid, int maxHistValue, int histPosition, int &momentumLeft, int &momentumRight, GetSetting laneRegionSetting) {
		int histLen = histImg.cols;
		int checkRange, numOfRegions;
		int extraPixel = 0;
		float momentRate;

		switch (histPosition) {
		case 1:
			checkRange = laneRegionSetting.checkRange1;
			momentRate = laneRegionSetting.moment1;
			break;

		case 2:
			checkRange = laneRegionSetting.checkRange2;
			momentRate = laneRegionSetting.moment2;

			break;

		case 3:
			checkRange = laneRegionSetting.checkRange3;
			momentRate = laneRegionSetting.moment3;

			break;

		case 4:
			checkRange = laneRegionSetting.checkRange4;
			momentRate = laneRegionSetting.moment4;
			break;

		default:
			break;
		}

		int minHistValue = maxHistValue >> 3;

		vector<HistRegion> histRegion;
		HistRegion tempRegion;

		int crrStartX, crrEndX, crrMidX, crrPeakValue, crrTotalValue;
		crrStartX = crrEndX = crrMidX = crrPeakValue = crrTotalValue = -1;

		// Calculate basic regions in histogram
		for (int i = 0; i < histLen; ++i) {
			if (histImg.at<float>(i) > 0) {
				if (crrStartX < 0) {
					crrStartX = i;
					crrTotalValue = crrPeakValue = histImg.at<float>(i);
				}
				else {
					crrTotalValue += histImg.at<float>(i);
					if (crrPeakValue < histImg.at<float>(i)) {
						crrPeakValue = histImg.at<float>(i);
					}
				}
			}
			else {
				if (crrTotalValue > minHistValue) {
					crrEndX = i - 1;
					crrMidX = (crrStartX + crrEndX) >> 1;
					tempRegion = HistRegion(crrStartX, crrEndX, crrMidX, crrPeakValue, crrTotalValue);
					histRegion.push_back(tempRegion);
					
					if (laneRegionSetting.debug) {
						cout << " Region " << " = " << tempRegion.startX << " -- " << tempRegion.midX << " -- " << tempRegion.endX << endl;
					}						
				}

				crrStartX = -1;
				crrTotalValue = -1;
			}
		}

		if (crrTotalValue > minHistValue) {
			crrEndX = histLen - 1;
			crrMidX = (crrStartX + crrEndX) >> 1;
			tempRegion = HistRegion(crrStartX, crrEndX, crrMidX, crrPeakValue, crrTotalValue);
			histRegion.push_back(tempRegion);
			
			if (laneRegionSetting.debug) {
				cout << " Region " << " = " << tempRegion.startX << " -- " << tempRegion.midX << " -- " << tempRegion.endX << endl;
			}
			
		}

		//if (laneRegionSetting.debug) {
		//	cout << " -- total value " << crrTotalValue << endl;
		//	cout << "==================================\n";
		//}
		

		if (histRegion.size() == 0) {
			momentumLeft = momentumRight = 0;
			return;
		}
		
		// Merge nearest regions
		mergeHistRegion(histRegion, maxHistValue);

		int minLeft, minRight, newLeft, newRight, leftRange1, leftRange2, rightRange1, rightRange2;
		int minLeft2, minRight2;

		newLeft = newRight = -1;

		leftRange1 = crrLeft + (float)momentumLeft * momentRate - checkRange;
		leftRange2 = crrLeft + (float)momentumLeft * momentRate + checkRange;
		rightRange1 = crrRight + (float)momentumRight * momentRate - checkRange;
		rightRange2 = crrRight + (float)momentumRight * momentRate + checkRange;

		minLeft = minRight = minLeft2 = minRight2 = histLen;

		//cout << "left 1 = " << crrLeft + leftRange1 << " -- left 2 = " << crrLeft + leftRange2;
		//cout << " -- right 1 = " << crrRight + rightRange1 << " -- right 2 = " << crrRight + rightRange2 << endl;

		//cout << "left 1 = " << leftRange1 << " -- left 2 = " << leftRange2;
		//cout << " -- right 1 = " << rightRange1 << " -- right 2 = " << rightRange2 << endl;

		int tempDist;
		float rateA, rateB;
		rateA = 0.4, rateB = 1 - rateA;

		for (int i = 0; i < histRegion.size(); ++i) {
			if (histRegion[i].midX < crrMid) {
				tempDist = abs(histRegion[i].midX - crrLeft) * rateA - (histRegion[i].peakValue / 2) * rateB;
				if (laneRegionSetting.debug) cout << " -- mid dis left = " << tempDist;
				if (tempDist < minLeft && histRegion[i].midX > leftRange1 && histRegion[i].midX < leftRange2) {
					minLeft = tempDist;
					newLeft = histRegion[i].midX;
					//cout << "min mid left = " << minLeft;
				}

				tempDist = abs(histRegion[i].startX - crrLeft) * rateA - (histRegion[i].peakValue) * rateB;
				if (laneRegionSetting.debug) cout << " -- start dis left = " << tempDist;
				if (tempDist < minLeft && histRegion[i].startX > leftRange1 && histRegion[i].startX < leftRange2) {
					minLeft = tempDist;
					newLeft = histRegion[i].startX;
					//cout << " -- min left left = " << minLeft;
				}

				tempDist = leftRange1 - histRegion[i].startX;
				if (tempDist < minLeft2) minLeft2 = tempDist;

				tempDist = leftRange1 - histRegion[i].midX;
				if (tempDist < minLeft2) minLeft2 = tempDist;

			}
			else {
				tempDist = abs(histRegion[i].midX - crrRight) * rateA - (histRegion[i].peakValue / 2) * rateB;
				if (laneRegionSetting.debug) cout << " -- mid dis right = " << tempDist;
				if (tempDist < minRight && histRegion[i].midX > rightRange1 && histRegion[i].midX < rightRange2) {
					minRight = tempDist;
					newRight = histRegion[i].midX;
					//cout << "min mid right = " << minRight;
				}
				
				tempDist = abs(histRegion[i].endX - crrRight) * rateA - (histRegion[i].peakValue) * rateB;
				if (laneRegionSetting.debug) cout << " -- end dis right = " << tempDist;
				if (tempDist < minRight && histRegion[i].endX > rightRange1 && histRegion[i].endX < rightRange2) {
					minRight = tempDist;
					newRight = histRegion[i].endX;
					//cout << " -- min mid right = " << minRight;
				}

				tempDist = histRegion[i].endX - rightRange2;
				if (tempDist < minRight2) minRight2 = tempDist;

				tempDist = histRegion[i].midX - rightRange2;
				if (tempDist < minRight2) minRight2 = tempDist;
			}

			if (laneRegionSetting.debug) {
				cout << " Region " << i << " = " << histRegion[i].startX << " -- " << histRegion[i].midX << " -- " << histRegion[i].endX << endl;
			}
		}

		if (laneRegionSetting.debug) cout << endl;
		
		if (laneRegionSetting.debug) cout << " min left = " << minLeft2 << " -- " << minRight2 << endl;

		if (newLeft > -1) {
			momentumLeft = newLeft - crrLeft;
			crrLeft = newLeft;
		}
		else if (minLeft2 < laneRegionSetting.checkRange5) {
			newLeft = leftRange1 - minLeft2;
			momentumLeft = newLeft - crrLeft;
			crrLeft = newLeft;
		}

		if (newRight > -1) {
			momentumRight = newRight - crrRight;
			crrRight = newRight;
		}	
		else if (minRight2 < laneRegionSetting.checkRange5) {
			newRight = rightRange2 + minRight2;
			momentumRight = newRight - crrRight;
			crrRight = newRight;
		}

		crrMid = (crrLeft + crrRight) >> 1;
	}
	
	//-------------------------------------------------------------------------------------------------//
	/**
	* This function merge too closed hist region in the set of
	& calculated input hist regions
	*/
	void mergeHistRegion(vector<HistRegion> &histRegion, int maxHist) {
		int crrHist = 0;
		int totalHist = histRegion.size();

		while (crrHist < totalHist - 1) {
			//cout << " -- " << crrHist;
			if (histRegion[crrHist + 1].startX - histRegion[crrHist].endX < 5 && histRegion[crrHist].totalValue < maxHist * 2) {
				histRegion[crrHist].endX = histRegion[crrHist + 1].endX;
				histRegion[crrHist].midX = (histRegion[crrHist].startX + histRegion[crrHist].endX) >> 1;
				histRegion[crrHist].peakValue = max(histRegion[crrHist].peakValue, histRegion[crrHist + 1].peakValue);
				histRegion[crrHist].totalValue += histRegion[crrHist + 1].totalValue;

				for (int i = crrHist + 1; i < histRegion.size() - 1; ++i) {
					histRegion[i] = histRegion[i + 1];
				}

				histRegion.pop_back();
				totalHist = histRegion.size();
				continue;
			}
			crrHist++;
		}
		//cout << " -- total hist = " << totalHist;
	}

	//-------------------------------------------------------------------------------------------------//
	/**
	* This function draw an image based on the input histogram
	* and input width and height
	*/
	void drawHistToImage(Mat inpHist, Mat &histImg, int width, int height, int maxHist, int scale) {
		histImg = Mat(height, width, CV_8UC3, Scalar(0, 0, 0));
		line(histImg, Point(inpHist.cols >> 1, 0), Point(inpHist.cols >> 1, height), Scalar(0, 0, 255), 1, 8, 0);
		line(histImg, Point(0, height - maxHist * scale), Point(inpHist.cols, height - maxHist * scale), Scalar(0, 255, 0), 1, 8, 0);

		for (int i = 0; i < inpHist.cols - 1; ++i) {
			line(histImg, Point(i, height - inpHist.at<float>(i) * scale), Point(i + 1, height - inpHist.at<float>(i + 1) * scale), Scalar(255, 0, 0), 2, 8, 0);
		}


	}

	//-------------------------------------------------------------------------------------------------//
	/**
	* This function return a histogram of amount of pixel within each column
	* from the ROI of the input image
	*/
	void getColumnHistFromROI(Mat &inpImg, int botRow, int topRow, Mat &histImg, int binSize) {
		int binHeight = topRow - botRow;
		int histWidth = inpImg.cols / binSize;
		histImg = Mat(1, histWidth, CV_32F);
		int crrX;

		for (int x = 0; x < histWidth; ++x) {
			crrX = x * binSize;
			histImg.at<float>(x) = sum(inpImg(Rect(crrX, botRow, min(binSize, inpImg.cols - crrX), binHeight)))[0] / 255;
		}
	}

	//-------------------------------------------------------------------------------------------------//
	/**
	* This function get the suitable IPM Matrix for the input image
	* \param inpImg: 3 channels color image
	* \param ipm: the ipm object which contains the transform matrix
	*/
	void getIPM(Mat &inpImg, IPM &ipm, GetSetting laneRegionSetting)
	{
		vector<Point2f> origPoints, dstPoints;
		Size size = inpImg.size();
		int width = size.width;
		int height = size.height;

		Size dstSize = Size(laneRegionSetting.bevWidth, laneRegionSetting.bevHeight);

		//--------- Get set of mapping points based on heuristic parameters ----
		int originBot = height;
		int originTop = height * laneRegionSetting.bevHeightOriginRate;
		int originLeftBot = width * laneRegionSetting.bevBotOriginRate;
		int originRightBot = width - originLeftBot;
		int originLeftTop = width * laneRegionSetting.bevTopOriginRate;

		int originRightTop = width - originLeftTop;

		//cout << laneRegionSetting.bevHeightOriginRate << " -- " << laneRegionSetting.bevBotOriginRate << endl;
		//cout << originBot << " " << originTop << " " << originLeftBot << " " << originRightBot << endl;

		int dstLeft = dstSize.width * laneRegionSetting.bevWidthMappingRate;
		int dstRight = dstSize.width - dstLeft;
		int dstTop = dstSize.height * laneRegionSetting.bevHeightMappingRate;
		int dstBot = dstSize.height;

		//------ Push mapping points to 2 set of points ----------
		origPoints.push_back(Point2f(originLeftBot, originBot));
		origPoints.push_back(Point2f(originRightBot, originBot));
		origPoints.push_back(Point2f(originRightTop, originTop));
		origPoints.push_back(Point2f(originLeftTop, originTop));


		dstPoints.push_back(Point2f(dstLeft, dstBot));
		dstPoints.push_back(Point2f(dstRight, dstBot));
		dstPoints.push_back(Point2f(dstRight, dstTop));
		dstPoints.push_back(Point2f(dstLeft, dstTop));

		ipm = IPM(size, dstSize, origPoints, dstPoints);

		//Mat tempImg;
		//if (laneRegionSetting.debug) {
		//	tempImg = inpImg.clone();
		//	ipm.getPoints(origPoints, dstPoints);
		//	ipm.drawPoints(origPoints, tempImg);
		//	imshow("Edited Input", tempImg);
		//	//imshow("Gray bev image", outImg);
		//}

		return;
	}

	//-------------------------------------------------------------------------------------------------//

}
