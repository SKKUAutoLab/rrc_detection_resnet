#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <fstream>

#include "GetSetting.h"

using namespace std;

GetSetting::GetSetting(){
}

void GetSetting::LaneRegion(char * fileName)
{
	ifstream fin;
	fin.open(fileName);
	char* buf_input;

	buf_input = new char[4096];
	fin.getline(buf_input, 4096);
	inputSequence = buf_input;
	inpWidth = atoi(inputSequence);

	buf_input = new char[4096];
	fin.getline(buf_input, 4096);
	inputSequence = buf_input;
	bevHeightOriginRate = atof(inputSequence);

	buf_input = new char[4096];
	fin.getline(buf_input, 4096);
	inputSequence = buf_input;
	bevTopOriginRate = atof(inputSequence);

	buf_input = new char[4096];
	fin.getline(buf_input, 4096);
	inputSequence = buf_input;
	bevBotOriginRate = atof(inputSequence);

	buf_input = new char[4096];
	fin.getline(buf_input, 4096);
	inputSequence = buf_input;
	bevWidthMappingRate = atof(inputSequence);

	buf_input = new char[4096];
	fin.getline(buf_input, 4096);
	inputSequence = buf_input;
	bevHeightMappingRate = atof(inputSequence);

	buf_input = new char[4096];
	fin.getline(buf_input, 4096);
	inputSequence = buf_input;
	bevWidth = atoi(inputSequence);

	buf_input = new char[4096];
	fin.getline(buf_input, 4096);
	inputSequence = buf_input;
	bevHeight = atoi(inputSequence);

	buf_input = new char[4096];
	fin.getline(buf_input, 4096);
	inputSequence = buf_input;
	meanHistogramFindingRate = atof(inputSequence);

	buf_input = new char[4096];
	fin.getline(buf_input, 4096);
	inputSequence = buf_input;
	minLenToFindSeedPoint = atoi(inputSequence);

	buf_input = new char[4096];
	fin.getline(buf_input, 4096);
	inputSequence = buf_input;
	edgeDetectionKernel = atoi(inputSequence);

	buf_input = new char[4096];
	fin.getline(buf_input, 4096);
	inputSequence = buf_input;
	numOfHistForGettingLanePoints = atoi(inputSequence);

	buf_input = new char[4096];
	fin.getline(buf_input, 4096);
	inputSequence = buf_input;
	outColorR = atoi(inputSequence);

	buf_input = new char[4096];
	fin.getline(buf_input, 4096);
	inputSequence = buf_input;
	outColorG = atoi(inputSequence);

	buf_input = new char[4096];
	fin.getline(buf_input, 4096);
	inputSequence = buf_input;
	outColorB = atoi(inputSequence);

	buf_input = new char[4096];
	fin.getline(buf_input, 4096);
	inputSequence = buf_input;
	checkRange1 = atoi(inputSequence);

	buf_input = new char[4096];
	fin.getline(buf_input, 4096);
	inputSequence = buf_input;
	checkRange2 = atoi(inputSequence);

	buf_input = new char[4096];
	fin.getline(buf_input, 4096);
	inputSequence = buf_input;
	checkRange3 = atoi(inputSequence);

	buf_input = new char[4096];
	fin.getline(buf_input, 4096);
	inputSequence = buf_input;
	checkRange4 = atoi(inputSequence);

	buf_input = new char[4096];
	fin.getline(buf_input, 4096);
	inputSequence = buf_input;
	checkRange5 = atoi(inputSequence);

	buf_input = new char[4096];
	fin.getline(buf_input, 4096);
	inputSequence = buf_input;
	moment1 = atof(inputSequence);

	buf_input = new char[4096];
	fin.getline(buf_input, 4096);
	inputSequence = buf_input;
	moment2 = atof(inputSequence);

	buf_input = new char[4096];
	fin.getline(buf_input, 4096);
	inputSequence = buf_input;
	moment3 = atof(inputSequence);

	buf_input = new char[4096];
	fin.getline(buf_input, 4096);
	inputSequence = buf_input;
	moment4 = atof(inputSequence);

	buf_input = new char[4096];
	fin.getline(buf_input, 4096);
	inputSequence = buf_input;
	debug = atof(inputSequence);

	fin.close();
}
