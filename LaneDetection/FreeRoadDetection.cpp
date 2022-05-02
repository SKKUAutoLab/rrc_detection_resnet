/*
FreeRoadDetection.cpp : The entry point for the lane detection module.

Core detection algorithm programmed by Tin Trung Duong.
File I/O Mechanism modification, refactoring, and version sync done by Hyung-Joon Jeon.

*/

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <time.h>
#include <stdlib.h>
#include <string>
#include <map>
#include <set>
#include <queue>
#include <sstream>
#include <thread>
#ifdef _WINDOWS_
#include <windows.h>
#define MAKEDIRS(o)	\
	struct _stat info; if (stat(o.c_str(), &info) != 0) { string mkdir_str = "mkdir -p "; mkdir_str.append(o); system(mkdir_str.c_str());	}
#define PATH_SEPARATOR '\\'
#define SLEEP(x) Sleep(x)
#else
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include <unistd.h>
#define MAKEDIRS(o)	\
	struct stat info; if (stat(o.c_str(), &info) != 0) { string mkdir_str = "mkdir -p "; mkdir_str.append(o); system(mkdir_str.c_str());	}
#define PATH_SEPARATOR '/'
#define SLEEP(x) usleep(x)
#endif
#include <iostream>

#pragma warning(disable : 4996)

using namespace std;
using namespace cv;

#include "GetSetting.h"
#include "IPM.h"
#include "SupportFunction.h"
#include "Preprocessing.h"
#include "MainFunction.h"
#include "EstimateLaneRegion.h"

using namespace SupportFunction;
using namespace Preprocessing;
using namespace MainFunction;
using namespace EstimateLaneRegion;

#define MAKEVIDEO_ENABLED 0

//=========== Global parameters ======================//

using NamedImageQueue = queue<pair<string, Mat>>;
NamedImageQueue queueImg;
//===================================================//

void static ReadImages(string inputDir, int inputType)
{
	vector<string> fileNameList;
	Mat inpImg;

	/* Open the directory */
#ifdef _WINDOWS_
	WIN32_FIND_DATA ffd;

	inputDir.append("\\*.png");
	HANDLE hFind = FindFirstFile(inputDir.c_str(), &ffd);

	if (hFind == INVALID_HANDLE_VALUE)
	{
		cerr << "Invalid handle value." << GetLastError() << endl;
		return;
	}

	bool finished = false;
	while (!finished)
	{
		fileNameList.push_back(string(ffd.cFileName));
		if (!FindNextFile(hFind, &ffd))
			finished = true;
	}
#else
	inputDir.erase(std::remove(inputDir.begin(), inputDir.end(), '\n'), inputDir.end());
	inputDir.erase(std::remove(inputDir.begin(), inputDir.end(), '\r'), inputDir.end());
	DIR *dir = opendir(inputDir.c_str());
	if (dir == NULL)
	{
		cerr << "opendir() failed. " << strerror(errno) << endl;
		return;
	}
	struct dirent *ent;
	while ((ent = readdir(dir)) != NULL)
	{
		string str = string(ent->d_name);

		if (str.find("png") < str.size())
			fileNameList.push_back(str);
	}
	closedir(dir);
#endif
	sort(fileNameList.begin(), fileNameList.end());
	for (int i = 0; i < fileNameList.size(); i++)
	{

		switch (inputType) {
		case 1: // image
			inpImg = imread(inputDir + PATH_SEPARATOR + fileNameList[i]);
			break;
		case 2: // video
			// TODO: implement case for video	
			inpImg = imread(inputDir + PATH_SEPARATOR + fileNameList[i]);
			break;
		default:
			cout << "Assuming image input." << endl;
			inpImg = imread(inputDir + PATH_SEPARATOR + fileNameList[i]);
			break;
		}
		queueImg.push(pair<string, Mat>({ fileNameList[i], inpImg }));
		while (queueImg.size() > 30) {
			SLEEP(50);
		}
	}
}

vector<string> split(string input, char delimiter) {
    vector<string> answer;
    stringstream ss(input);
    string temp;
 
    while (getline(ss, temp, delimiter)) {
        answer.push_back(temp);
    }
 
    return answer;
}

int main()
{
	string inputDir;
	string outputDir;
	int inputType;
	int outputType;

	/* Get I/O parameters */
	ifstream fin((char *)"LaneIOSetting.txt");
	getline(fin, inputDir);
	getline(fin, outputDir);
	fin >> inputType;
	fin >> outputType;
	fin.close();

	/* Check if output directory exists and make one if it doesn't */
	MAKEDIRS(outputDir)

	int crrIndex = 0;
	Mat inpImg, outImg;
	string outputFilePath;
#if MAKEVIDEO_ENABLED
	VideoWriter out_capture;
	if (outputType == 2) {
		outputFilePath = outputDir + PATH_SEPARATOR + "output.mp4";
		out_capture = VideoWriter(outputFilePath, CV_FOURCC('M', 'J', 'P', 'G'), 1, Size(640, 336));
	}
#endif
	clock_t begin;
	double total_secs = 0.0;

	/* Start a thread to read and stack a sequence of input images */
	thread readWorkHorse(ReadImages, inputDir, inputType);

	GetSetting laneRegionSetting;
	laneRegionSetting.LaneRegion((char *)"LaneRegionSetting.txt");

	cout << "[Lane Detection] Reading input images..." << endl;

	while (crrIndex < 7581)
	{
		if (!queueImg.empty())
		{
			pair<string, Mat> crrImgPair = queueImg.front();
			cout << "--------------- Frame name " << crrImgPair.first << " --------" << endl;
			inpImg = crrImgPair.second;
			queueImg.pop();

			begin = clock();
			getEstimateLaneRegion(inpImg, outImg, laneRegionSetting);
			total_secs += double(clock() - begin) / CLOCKS_PER_SEC;

			if (outputType == 1) {
				/* Although redundant, the following is needed since Lane Module should work within pipeline */
				imwrite(outputDir + PATH_SEPARATOR + crrImgPair.first, inpImg);

				outputFilePath = outputDir + PATH_SEPARATOR + split(crrImgPair.first, '.')[0] + ".bmp";
				imwrite(outputFilePath, outImg);
			}
#if MAKEVIDEO_ENABLED
			else if (outputType == 2) {
				out_capture.write(outImg);
			}
#endif
			waitKey(1 - laneRegionSetting.debug);
			crrIndex++;
		}
		else
		{
			SLEEP(30);
		}
	}
	readWorkHorse.join();

	cout << "FPS:" << crrIndex / total_secs << endl;
	system("pause");
	return 0;
}
