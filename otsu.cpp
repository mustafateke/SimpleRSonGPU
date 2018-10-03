////try this
////----------------------------------------------------------------
//// binarisation otsu
////---------------------------------------------------------------
//#include <iostream>
//#include <stdio.h>
//#include <stdlib.h>
//#include <math.h>
//#include <time.h>
//#include "opencv/cxcore.h"
//#include "opencv/cv.h"
//#include "highgui.h"
//
//int GRAYLEVEL = 256;
//#define MAX_BRIGHTNESS 255
////const double INFO_THRESHOLD = 0.2;
//
//using namespace std;
//
////----------------------------------------------------------------------------
//// binarization by Otsu's method
//// based on maximization of inter-class variance
////----------------------------------------------------------------------------
//void binarize_otsu(char* pData, int height, int width, int size )
//{
//	int hist[GRAYLEVEL];
//	double prob[GRAYLEVEL],
//		omega[GRAYLEVEL]; // prob of graylevels
//	double myu[GRAYLEVEL]; // mean value for separation
//	double max_sigma,
//		sigma[GRAYLEVEL]; // inter-class variance
//
//	int i, x, y; /* Loop variable */
//	int threshold; /* threshold for binarization */
//
//	// Histogram generation
//	memset((int*) hist , 0, GRAYLEVEL * sizeof(int) );
//
//	CvSize size = cvGetSize(image);
//
//	for (int i = 0; i < height; ++i)
//	{
//		unsigned char* pData = (unsigned char*) (image->imageData + i *
//			image->widthStep);
//		for (int j = 0; j < width; ++j)
//		{
//			int k = (int)((unsigned char) *(pData+j));
//			hist[k]++;
//		}
//	}
//
//	// calculation of probability density
//	for ( i = 0; i < GRAYLEVEL; ++i )
//	{
//		prob[i] = (double) ((double)hist[i] / (double)size);
//	}
//
//	// omega & myu generation
//	omega[0] = prob[0];
//	myu[0] = 0.0;
//	for (i = 1; i < GRAYLEVEL; i++)
//	{
//		omega[i] = omega[i-1] + prob[i];
//		myu[i] = myu[i-1] + (i*prob[i]);
//	}
//
//	//-----------------------------------------------------------------
//	// sigma maximization
//	// sigma stands for inter-class variance
//	// and determines optimal threshold value
//	//----------------------------------------------------------------
//	threshold = 0;
//	max_sigma = 0.0;
//	for (i = 0; i < GRAYLEVEL-1; i++)
//	{
//		if (omega[i] != 0.0 && omega[i] != 1.0)
//		{
//			//sigma[i] = (omega[i]*(1.0 - omega[i])) * ((myu[GRAYLEVEL-1] - 2*myu[i]) *
//			(myu[GRAYLEVEL-1] - 2*myu[i]));
//			sigma[i] = ((myu[GRAYLEVEL-1]*omega[i] - myu[i]) *
//				(myu[GRAYLEVEL-1]*omega[i] - myu[i])) / (omega[i]*(1.0 - omega[i]));
//		}
//		else
//		{
//			sigma[i] = 0.0;
//		}
//		if (sigma[i] > max_sigma)
//		{
//			max_sigma = sigma[i];
//			threshold = i;
//		}
//	}
//}
//
//
