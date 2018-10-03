/*
* Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/
#include "stdafx.h"
#ifndef _SIMPLETEXTURE_KERNEL_H_
#define _SIMPLETEXTURE_KERNEL_H_

#if __CUDA_ARCH__ > 100      // Atomics only used with > sm_10 architecture
#include <sm_20_atomic_functions.h>
#endif

#define PI  3.14159
#define GREY_LEVEL 255
#define GRAYLEVEL 256
#define BIGNUMBER 100000000

#define SQUARE(a)  ((a)*(a))
#define CARRE(X) ((X)*(X))
#define MAX(a,b)   (((a)>(b))?(a):(b))
#define MIN(a,b)   (((a)>(b))?(b):(a))

// declare texture reference for 2D float texture
typedef char ValueType;


// declare texture reference for 2D float texture
texture<ValueType, 1, cudaReadModeElementType> imgTex;

////////////////////////////////////////////////////////////////////////////////
//! Transform an image using texture lookups
//! @param pImgData  output data in global memory
////////////////////////////////////////////////////////////////////////////////

__global__ void
RGBShadowKernel( ValueType * pImgData, char *pNDVI, char* pShadow, int width, int height,int channels, int step, int maskWidthStep) 
{
	// calculate normalized texture coordinates
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	// read from texture and write to global memory
	//for(i=0;i<height;i++) for(j=0;j<width;j++) for(k=0;k<channels;k++)


	if(x < width && y < height)
	{
		float fRed, fGreen, fBlue;

		float fHue, fSat,fValue;

		fRed	= pImgData[y*step+x*channels+1]/255.0;
		fGreen	= pImgData[y*step+x*channels+2]/255.0;
		fBlue	= pImgData[y*step+x*channels+3]/255.0;

		float maxRGB=MAX(MAX(fRed,fGreen),MAX(fGreen,fBlue));
		float minRGB=MIN(MIN(fRed,fGreen),MIN(fGreen,fBlue));

		fValue=maxRGB;                                                

		if (maxRGB>0.0)   (fSat)= (maxRGB-minRGB)/maxRGB;     
		else                            (fSat)=0.0;


		if ((fSat)==0)     (fHue)=0.0;                                              
		else
		{ 
			if (fRed==maxRGB)		(fHue)=		(fGreen-fGreen)/(maxRGB-minRGB);    
			else if (fGreen==maxRGB)(fHue)=2+	(fBlue-fRed)/(maxRGB-minRGB);
			else					(fHue)=4+	(fRed-fGreen)/(maxRGB-minRGB);

			(fHue)*=60;           
			if ((fHue)<0.0)			(fHue)+=360;  
		}

		pShadow[y*(maskWidthStep)+x] = 255*(fValue-fSat<-0.15);
		/*pNDVI*/

		//printf("RGB[%f %f %f]", fRed, fGreen, fBlue);
		//printf("RGB[%f %f %f]->HSV[[%f %f %f]\n", fRed, fGreen, fBlue, fHue, fSat, fValue);
		/*pImgData[y*step+x*channels+4]=50;*/
	}
	//  pImgData[y*width + x] = tex1D(imgTex, x);
}

__global__ void
RGBNIRShadowKernel( ValueType * pImgData, char *pNDVI, char* pShadow, int width, int height,int channels, int step, int maskWidthStep) 
{
	// calculate normalized texture coordinates
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	// read from texture and write to global memory
	//for(i=0;i<height;i++) for(j=0;j<width;j++) for(k=0;k<channels;k++)


	if(x < width && y < height)
	{
		float fRed, fGreen, fBlue, fNIR;

		float fHue, fSat,fValue;

		fRed	= (unsigned char)pImgData[y*step+x*channels+0];
		fGreen	= (unsigned char)pImgData[y*step+x*channels+1];
		fBlue	= (unsigned char)pImgData[y*step+x*channels+2];
		//fNIR	= (unsigned char)pImgData[y*step+x*channels+4];

		unsigned char maxRGB=MAX(MAX(fRed,fGreen),MAX(fGreen,fBlue));
		unsigned char minRGB=MIN(MIN(fRed,fGreen),MIN(fGreen,fBlue));

		fValue=maxRGB;                                                

		if (maxRGB>0.0){/*printf("maxRGB: %d ", maxRGB); */  (fSat)= (maxRGB-minRGB)/((double)maxRGB); }    
		else{     (fSat)=0.0;}


		if ((fSat)==0)     (fHue)=0.0;                                              
		else
		{ 
			if (fRed==maxRGB)		(fHue)=		(fGreen-fGreen)/(maxRGB-minRGB);    
			else if (fGreen==maxRGB)(fHue)=2+	(fBlue-fRed)/(maxRGB-minRGB);
			else					(fHue)=4+	(fRed-fGreen)/(maxRGB-minRGB);

			(fHue)*=60;           
			if ((fHue)<0.0)			(fHue)+=360;  
		}

		pShadow[y*(maskWidthStep)+x] = 255*(fValue/255.0 - fSat<-0.05);
		/*pNDVI*/

		//printf("RGB[%f %f %f]", fRed, fGreen, fBlue);
		//printf("RGB[%f %f %f]->HSV[[%f %f %f]\n", fRed, fGreen, fBlue, fHue, fSat, fValue);
		/*pImgData[y*step+x*channels+4]=50;*/
	}
	//  pImgData[y*width + x] = tex1D(imgTex, x);
}

void
NDVIKernel( ValueType * pImgData, char *pNDVI, int* pHistograms, int width, int height,int channels, int step, int maskWidthStep) 
{
#pragma omp parallel for
	for(int x = 0; x<width; x++)
#pragma omp parallel for
		for(int y = 0; y < height; y++)
		{
			double fRed, fGreen, fBlue, fNIR;
			short *pData = (short *)(pImgData);
			fRed	= /*(unsigned char)*/pData[y*step/2+x*channels+0];
			//fGreen	= (unsigned char)pImgData[y*step+x*channels+2];
			//fBlue	= (unsigned char)pImgData[y*step+x*channels+3];
			fNIR	= /*(unsigned char)*/pData[y*step/2+x*channels+3];
			float ndvi = (fNIR-fRed)/(fNIR+fRed);
			int nValue = 255*(( 1+ndvi )/2.0);
			if(nValue> 255) nValue = 255;
			if(nValue < 0) nValue = 0;

			pNDVI[y*(maskWidthStep)+x] = nValue;
			//if(nValue < 0){printf("%d ", nValue);}

			pHistograms[nValue] += 1;
			//atomicAdd(&pHistograms[nValue], 1);
			/*pNDVI*/

			//printf("RGB[%f %f %f]", fRed, fGreen, fBlue);
			//printf("RGB[%f %f %f]->HSV[[%f %f %f]\n", fRed, fGreen, fBlue, fHue, fSat, fValue);
			/*pImgData[y*step+x*channels+4]=50;*/
		}
		//  pImgData[y*width + x] = tex1D(imgTex, x);
}

__global__ void
NDVIKernelOtsu( ValueType * pImgData, char *pNDVI, char* pShadow,int *pHistograms, int width, int height, int size, int channels, int step, int maskWidthStep) 
{
	// calculate normalized texture coordinates
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	__shared__  int threshold; /* threshold for binarization */

	// read from texture and write to global memory
	//for(i=0;i<height;i++) for(j=0;j<width;j++) for(k=0;k<channels;k++)


	if(x < width && y < height)
	{
		int hist[GRAYLEVEL];
		float prob[GRAYLEVEL],
			omega[GRAYLEVEL]; // prob of graylevels
		float myu[GRAYLEVEL]; // mean value for separation
		float max_sigma,
			sigma[GRAYLEVEL]; // inter-class variance

		int i; /* Loop variable */

		//if(x==0 && y == 0)
		{
			// calculation of probability density
			for ( i = 0; i < GRAYLEVEL; ++i )
			{
				prob[i] = (double) ((double)pHistograms[i] / (double)size);
			}

			// omega & myu generation
			//__syncthreads();
			omega[0] = prob[0];
			myu[0] = 0.0;
			for (i = 1; i < GRAYLEVEL; i++)
			{
				omega[i] = omega[i-1] + prob[i];
				myu[i] = myu[i-1] + (i*prob[i]);
			}

			//-----------------------------------------------------------------
			// sigma maximization
			// sigma stands for inter-class variance
			// and determines optimal threshold value
			//----------------------------------------------------------------
			threshold = 0;
			max_sigma = 0.0;
			for (i = 0; i < GRAYLEVEL-1; i++)
			{
				if (omega[i] != 0.0 && omega[i] != 1.0)
				{
					//sigma[i] = (omega[i]*(1.0 - omega[i])) * ((myu[GRAYLEVEL-1] - 2*myu[i]) *
					//(myu[GRAYLEVEL-1] - 2*myu[i]));
					sigma[i] = ((myu[GRAYLEVEL-1]*omega[i] - myu[i]) *
						(myu[GRAYLEVEL-1]*omega[i] - myu[i])) / (omega[i]*(1.0 - omega[i]));
				}
				else
				{
					sigma[i] = 0.0;
				}
				if (sigma[i] > max_sigma)
				{
					max_sigma = sigma[i];
					threshold = i;
				}
			}
		}
		//__syncthreads();
		//pNDVI[y*(maskWidthStep)+x] = 255*(NDVIndex>threshold?1:0);


	}
	//  pImgData[y*width + x] = tex1D(imgTex, x);
}

void
CalculateThreshold( int *pHistograms,
				   int width, int height, int size, int *pThreshold) 
{

	int threshold; /* threshold for binarization */

	// read from texture and write to global memory
	//for(i=0;i<height;i++) for(j=0;j<width;j++) for(k=0;k<channels;k++)



	int hist[GRAYLEVEL];
	float prob[GRAYLEVEL],
		omega[GRAYLEVEL]; // prob of graylevels
	float myu[GRAYLEVEL]; // mean value for separation
	float max_sigma,
		sigma[GRAYLEVEL]; // inter-class variance

	int i; /* Loop variable */

	// calculation of probability density
	for ( i = 0; i < GRAYLEVEL; ++i )
	{
		prob[i] = (double) ((double)pHistograms[i] / (double)size);
	}

	// omega & myu generation
	//__syncthreads();
	omega[0] = prob[0];
	myu[0] = 0.0;
	for (i = 1; i < GRAYLEVEL; i++)
	{
		omega[i] = omega[i-1] + prob[i];
		myu[i] = myu[i-1] + (i*prob[i]);
	}

	//-----------------------------------------------------------------
	// sigma maximization
	// sigma stands for inter-class variance
	// and determines optimal threshold value
	//----------------------------------------------------------------
	threshold = 0;
	max_sigma = 0.0;
	for (i = 0; i < GRAYLEVEL-1; i++)
	{
		if (omega[i] != 0.0 && omega[i] != 1.0)
		{
			//sigma[i] = (omega[i]*(1.0 - omega[i])) * ((myu[GRAYLEVEL-1] - 2*myu[i]) *
			//(myu[GRAYLEVEL-1] - 2*myu[i]));
			sigma[i] = ((myu[GRAYLEVEL-1]*omega[i] - myu[i]) *
				(myu[GRAYLEVEL-1]*omega[i] - myu[i])) / (omega[i]*(1.0 - omega[i]));
		}
		else
		{
			sigma[i] = 0.0;
		}
		if (sigma[i] > max_sigma)
		{
			max_sigma = sigma[i];
			threshold = i;
		}
	}

	*pThreshold =  threshold;

}


void
CalculateHistogram( ValueType * pImgData, int *pHistograms, int width, int height, int size, int step) 
{
#pragma omp parallel for
	for(int x = 0; x<width; x++)
#pragma omp parallel for
		for(int y = 0; y < height; y++)
		{
			unsigned char nValue;
			nValue	= (unsigned char)pImgData[y*step+x];
			pHistograms[nValue] += 1;
			//atomicAdd(&pHistograms[nValue], 1);
		}

}

void
Threshold( ValueType * pImgData, int threshold, int width, int height, int size, int step) 
{
#pragma omp parallel for
	for(int x = 0; x<width; x++)
#pragma omp parallel for
		for(int y = 0; y < height; y++)
		{
			int nValue;
			nValue	=(unsigned char)pImgData[y*step+x];
			pImgData[y*step+x] = 255*(nValue>threshold?1:0);
		}

}
__global__ void
RGBNIRShadowNDVIKernel( ValueType * pImgData, char *pNDVI, char* pShadow, int width, int height,int channels, int step, int maskWidthStep) 
{
	// calculate normalized texture coordinates
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	// read from texture and write to global memory
	//for(i=0;i<height;i++) for(j=0;j<width;j++) for(k=0;k<channels;k++)

	if(x < width && y < height)
	{
		float fRed, fGreen, fBlue, fNIR;

		float fHue, fSat,fValue;

		fRed	= (unsigned char)pImgData[y*step+x*channels+1];
		fGreen	= (unsigned char)pImgData[y*step+x*channels+2];
		fBlue	= (unsigned char)pImgData[y*step+x*channels+3];
		fNIR	= (unsigned char)pImgData[y*step+x*channels+4];

		unsigned char maxRGB=MAX(MAX(fRed,fGreen),MAX(fGreen,fBlue));
		unsigned char minRGB=MIN(MIN(fRed,fGreen),MIN(fGreen,fBlue));

		fValue=maxRGB;                                                

		if (maxRGB>0.0){/*printf("maxRGB: %d ", maxRGB); */  (fSat)= (maxRGB-minRGB)/((double)maxRGB); }    
		else{     (fSat)=0.0;}


		if ((fSat)==0)     (fHue)=0.0;                                              
		else
		{ 
			if (fRed==maxRGB)		(fHue)=		(fGreen-fGreen)/(maxRGB-minRGB);    
			else if (fGreen==maxRGB)(fHue)=2+	(fBlue-fRed)/(maxRGB-minRGB);
			else					(fHue)=4+	(fRed-fGreen)/(maxRGB-minRGB);

			(fHue)*=60;           
			if ((fHue)<0.0)			(fHue)+=360;  
		}

		pShadow[y*(maskWidthStep)+x] = 255*(fValue/255.0 - fSat<-0.35);

		pNDVI[y*(maskWidthStep)+x] = 255*(( (fNIR-fRed)/(fNIR+fRed)) /*> 0.0*/);
		/*pNDVI*/

		//printf("RGB[%f %f %f]", fRed, fGreen, fBlue);
		//printf("RGB[%f %f %f]->HSV[[%f %f %f]\n", fRed, fGreen, fBlue, fHue, fSat, fValue);
		/*pImgData[y*step+x*channels+4]=50;*/
	}
	//  pImgData[y*width + x] = tex1D(imgTex, x);
}

__global__ void
HSVShadowKernel( ValueType * pImgData, char *pNDVI, char* pShadow, int width, int height,int channels, int step, int maskWidthStep) 
{
	// calculate normalized texture coordinates
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	// read from texture and write to global memory
	//for(i=0;i<height;i++) for(j=0;j<width;j++) for(k=0;k<channels;k++)


	if(x < width && y < height)
	{
		float fRed, fGreen, fBlue, fNIR;

		float fHue, fSat,fValue;

		fHue	= pImgData[y*step+x*channels+1]/255.0;
		fSat	= pImgData[y*step+x*channels+2]/255.0;
		fValue	= pImgData[y*step+x*channels+3]/255.0;


		pShadow[y*(maskWidthStep)+x] = 255*(fValue-fSat<-0.05);
		/*pNDVI*/

		//printf("RGB[%f %f %f]", fRed, fGreen, fBlue);
		//printf("RGB[%f %f %f]->HSV[[%f %f %f]\n", fRed, fGreen, fBlue, fHue, fSat, fValue);
		/*pImgData[y*step+x*channels+4]=50;*/
	}
	//  pImgData[y*width + x] = tex1D(imgTex, x);
}

__global__ void
MinMaxKernel( ValueType * pImgData,
			 char *pNDVI, 
			 char* pShadow,
			 int *pHistograms,
			 int width, int height,int channels, int step, int maskWidthStep) 
{
	// calculate normalized texture coordinates
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	// read from texture and write to global memory
	//for(i=0;i<height;i++) for(j=0;j<width;j++) for(k=0;k<channels;k++)


	if(x < width && y < height)
	{
		unsigned char fRed, fGreen, fBlue, fNIR;
		float fHue, fSat,fValue;
		__shared__ unsigned char minRed, minGreen, minBlue, minNIR;
		__shared__ unsigned char maxRed, maxGreen, maxBlue, maxNIR;

		fRed	= pImgData[y*step+x*channels+0];
		fGreen	= pImgData[y*step+x*channels+1];
		fBlue	= pImgData[y*step+x*channels+2];
		fNIR	= pImgData[y*step+x*channels+3];



		atomicAdd(&pHistograms[fRed], 1);
		atomicAdd(&pHistograms[256 + fGreen], 1);
		atomicAdd(&pHistograms[512 + fBlue], 1);
		atomicAdd(&pHistograms[768 + fNIR], 1);

		__syncthreads();

		/*if(threadIdx.x==0)*/{
			for(int i=0; i< 255; i++){
				if(pHistograms[i] != 0){minRed = i; break;}
			}

			for(int i=255; i> 0; i--){
				if(pHistograms[i] != 0){maxRed = i; break;}
			}
			//printf("minred: %d ", minRed);
			//printf("maxred: %d\n", maxRed);
			//Green
			for(int i=0; i< 255; i++){
				if(pHistograms[i+256 ] != 0){minGreen = i; break;}
			}

			for(int i=255; i> 0; i--){
				if(pHistograms[i+256 ] != 0){maxGreen = i; break;}
			}
			//printf("minGreen: %d ", minGreen);
			//printf("maxGreen: %d\n", maxGreen);

			for(int i=0; i< 255; i++){
				if(pHistograms[i+512] != 0){minBlue = i; break;}
			}

			for(int i=255; i> 0; i--){
				if(pHistograms[i+512] != 0){maxBlue = i; break;}
			}
			//printf("minBlue: %d ", minBlue);
			//printf("maxBlue: %d\n", maxBlue);

			for(int i=0; i< 255; i++){
				if(pHistograms[i+768] != 0){minNIR = i; break;}
			}

			for(int i=255; i> 0; i--){
				if(pHistograms[i+768] != 0){maxNIR = i; break;}
			}
			//printf("minNIR: %d ", minNIR);
			//printf("maxNIR: %d\n", maxNIR);
		}
		//__syncthreads();
		//fRed = (unsigned char)( (fRed-minRed)*255.0/(maxRed-minRed) );
		//fGreen = (unsigned char)( (fGreen-minGreen)*255.0/(maxGreen-minGreen) );
		//fBlue = (unsigned char)( (fBlue-minBlue)*255.0/(maxBlue-minBlue) );
		//fNIR = (unsigned char)( (fNIR-minNIR)*255.0/(maxNIR-minNIR) );
		//__syncthreads();

		//unsigned char maxRGB=MAX(MAX(fRed,fGreen),MAX(fGreen,fBlue));
		//unsigned char minRGB=MIN(MIN(fRed,fGreen),MIN(fGreen,fBlue));

		//fValue=maxRGB;                                                

		//if (maxRGB>0.0){/*printf("maxRGB: %d ", maxRGB); */  (fSat)= (maxRGB-minRGB)/((double)maxRGB); }    
		//else{     (fSat)=0.0;}


		//if ((fSat)==0)     (fHue)=0.0;                                              
		//else
		//{ 
		//	if (fRed==maxRGB)		(fHue)=		(fGreen-fGreen)/(maxRGB-minRGB);    
		//	else if (fGreen==maxRGB)(fHue)=2+	(fBlue-fRed)/(maxRGB-minRGB);
		//	else					(fHue)=4+	(fRed-fGreen)/(maxRGB-minRGB);

		//	(fHue)*=60;           
		//	if ((fHue)<0.0)			(fHue)+=360;  
		//}

		//pShadow[y*(maskWidthStep)+x] = 255*(fValue/255.0 - fSat<-0.35);

		//pNDVI[y*(maskWidthStep)+x] = 255*(( (fNIR-fRed)/(fNIR+fRed)) /*> 0.0*/);

		//float fRed, fGreen, fBlue, fNIR;

		//pShadow[y*(maskWidthStep)+x] = 255*(( ((double)fNIR-fRed)/(fNIR+fRed)) /*> 0.22*/);
		/*pNDVI*/

		//printf("RGB[%f %f %f]", fRed, fGreen, fBlue);
		//printf("RGB[%f %f %f]->HSV[[%f %f %f]\n", fRed, fGreen, fBlue, fHue, fSat, fValue);
		/*pImgData[y*step+x*channels+4]=50;*/


		/*pNDVI*/

		//printf("RGB[%f %f %f]", fRed, fGreen, fBlue);
		//printf("RGB[%f %f %f]->HSV[[%f %f %f]\n", fRed, fGreen, fBlue, fHue, fSat, fValue);
		/*pImgData[y*step+x*channels+4]=50;*/
	}
	//  pImgData[y*width + x] = tex1D(imgTex, x);
}

void
CalculateHistograms( ValueType * pImgData,
					int *pHistograms,
					int width, int height,int channels, int step, int max) 
{
#pragma omp parallel for
	for(int x = 0; x<width; x++)
#pragma omp parallel for
		for(int y = 0; y < height; y++)
		{
			short fRed, fGreen, fBlue, fNIR;
			float fHue, fSat,fValue;


			short *pData = (short *)(pImgData);
			fRed	= pData[y*step/2+x*channels+0];
			fGreen	= pData[y*step/2+x*channels+1];
			fBlue	= pData[y*step/2+x*channels+2];
			fNIR	= pData[y*step/2+x*channels+3];



			pHistograms[fRed]+=1; //atomicAdd(&pHistograms[fRed], 1);
			pHistograms[max + fGreen]+=1; //atomicAdd(&pHistograms[max + fGreen], 1);
			pHistograms[2*max + fBlue] +=1; //atomicAdd(&pHistograms[2*max + fBlue], 1);
			pHistograms[3*max + fNIR]+=1; //atomicAdd(&pHistograms[3*max + fNIR], 1);

		}
		//  pImgData[y*width + x] = tex1D(imgTex, x);
}

void
StretchImage( ValueType * pImgData,
			 int *pHistograms,
			 int width, int height,int channels, int step,int max) 
{

	short minRed, minGreen, minBlue, minNIR;
	short maxRed, maxGreen, maxBlue, maxNIR;
	// read from texture and write to global memory
	//for(i=0;i<height;i++) for(j=0;j<width;j++) for(k=0;k<channels;k++)

#pragma omp parallel for
	for(int x = 0; x<width; x++)
#pragma omp parallel for
		for(int y = 0; y < height; y++)
		{
			for(int i=0; i< max; i++){
				if(pHistograms[i] != 0){minRed = i; break;}
			}

			for(int i=max; i> 0; i--){
				if(pHistograms[i] != 0){maxRed = i; break;}
			}

			for(int i=0; i< max; i++){
				if(pHistograms[i+max ] != 0){minGreen = i; break;}
			}

			for(int i=max; i> 0; i--){
				if(pHistograms[i+max ] != 0){maxGreen = i; break;}
			}


			for(int i=0; i< max; i++){
				if(pHistograms[i+2*max] != 0){minBlue = i; break;}
			}

			for(int i=max; i> 0; i--){
				if(pHistograms[i+2*max] != 0){maxBlue = i; break;}
			}


			for(int i=0; i< max; i++){
				if(pHistograms[i+3*max] != 0){minNIR = i; break;}
			}

			for(int i=max; i> 0; i--){
				if(pHistograms[i+3*max] != 0){maxNIR = i; break;}
			}

			short fRed, fGreen, fBlue, fNIR;
			short *pData = (short *)(pImgData);
			fRed	= pData[y*step/2+x*channels+0];
			fGreen	= pData[y*step/2+x*channels+1];
			fBlue	=pData[y*step/2+x*channels+2];
			fNIR	= pData[y*step/2+x*channels+3];

			pData[y*step/2+x*channels+0] = (short)( (fRed-minRed)*2048.0/(maxRed-minRed) );
			pData[y*step/2+x*channels+1] = (short)( (fGreen-minGreen)*2048.0/(maxGreen-minGreen) );
			pData[y*step/2+x*channels+2] = (short)( (fBlue-minBlue)*2048.0/(maxBlue-minBlue) );
			pData[y*step/2+x*channels+3] = (short)( (fNIR-minNIR)*2048.0/(maxNIR-minNIR) );

		}


}

void
CalculateCDF( int *pHistograms, int *pCDFHistograms,
			 int width, int height, int size,  int max, int channels) 
{

	pCDFHistograms[0] = pHistograms[0];
	pCDFHistograms[max+0] = pHistograms[max+0];
	pCDFHistograms[2*max+0] = pHistograms[2*max+0];
	pCDFHistograms[3*max+0] = pHistograms[3*max+0];
	for (int i = 1; i < max; i++)
	{
		for (int channel = 0; channel< channels; channel++)
		{
			pCDFHistograms[channel*max+i] = pCDFHistograms[channel*max+i - 1] + pHistograms[channel*max+i] ;
		}
	}
}

void
HistEqImage( ValueType * pImgData,
			int *pHistograms,
			int width, int height,int channels, int step,int max) 
{

	short minRed, minGreen, minBlue, minNIR;
	short maxRed, maxGreen, maxBlue, maxNIR;
	// read from texture and write to global memory
	//for(i=0;i<height;i++) for(j=0;j<width;j++) for(k=0;k<channels;k++)

#pragma omp parallel for
	for(int x = 0; x<width; x++)
#pragma omp parallel for
		for(int y = 0; y < height; y++)
		{

			unsigned short fRed, fGreen, fBlue, fNIR;
			short *pData = (short *)(pImgData);
			fRed		= pData[y*step/2+x*channels+0];
			fGreen	= pData[y*step/2+x*channels+1];
			fBlue		= pData[y*step/2+x*channels+2];
			fNIR		= pData[y*step/2+x*channels+3];
			int size= (width*height);

			//if(max*0+ fRed == 2046){
			//	printf(" val[%d]: %d, size: %d\n", fRed,(unsigned int)(max-1)*pHistograms[max*0+ fRed ]/size, (int)size );
			//	printf(" max: %d\n",  max);
			//	printf(" cdf: %d\n",pHistograms[max*0+ fRed ]);
			//}
			pData[y*step/2+x*channels+0] = (unsigned short)( (max-1)*pHistograms[max*0+ fRed ]/size);// (short)( pHistograms[max*0+ fRed ]*alpha );
			pData[y*step/2+x*channels+1] = (unsigned short)( (max-1)*pHistograms[max*1+ fGreen]/size );
			pData[y*step/2+x*channels+2] = (unsigned short)( (max-1)*pHistograms[max*2+ fBlue]/size );
			pData[y*step/2+x*channels+3] = (unsigned short)( (max-1)*pHistograms[max*3+ fNIR ]/size );

		}

}
#endif // #ifndef _SIMPLETEXTURE_KERNEL_H_
