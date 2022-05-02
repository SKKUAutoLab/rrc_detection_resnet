#ifndef __GETSETTING_H__
#define __GETSETTING_H__

class GetSetting
{
public:
	char* inputSequence;
	
	int inpWidth;
	float bevHeightOriginRate;
	float bevTopOriginRate;
	float bevBotOriginRate;
	float bevWidthMappingRate;
	float bevHeightMappingRate;
	int bevWidth;
	int bevHeight;

	float meanHistogramFindingRate;

	int minLenToFindSeedPoint;
	int edgeDetectionKernel;
	int numOfHistForGettingLanePoints;

	int outColorR;
	int outColorG;
	int outColorB;

	int checkRange1;
	int checkRange2;
	int checkRange3;
	int checkRange4;
	int checkRange5;

	float moment1;
	float moment2;
	float moment3;
	float moment4;

	int debug;

	GetSetting();
	void LaneRegion(char* fileName);

private:

};

#endif /*__GETSETTING_H__*/