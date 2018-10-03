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

/*
* This sample demonstrates how use texture fetches in CUDA
*
* This sample takes an input PGM image (image_filename) and generates 
* an output PGM image (image_filename_out).  This CUDA kernel performs
* a simple 2D transform (rotation) on the texture coordinates (u,v).
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <math.h>

// includes, project
#include <cutil_inline.h>

// includes, kernels
#include <Kernel.cu>

char *image_filename = "lena_bw.pgm";
char *ref_filename   = "ref_rotated.pgm";
float angle = 0.5f;    // angle to rotate image by (in radians)

#define MIN_EPSILON_ERROR 5e-3f

int xLoadImage();
int GeoTiffRead(const char *fileName, int dataSize = 8, int channels = 3);
int GeoTiffShow();
typedef char ValueType;
extern int nWidth;
extern int nHeight ;
extern int nChannels;
extern int nWidthStep;
extern int nMaskWidthStep;
extern ValueType * pData ;
extern char * pShadowMask ;
extern char * pNDVIMask ;

cudaArray* cuArray;

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest( );


void loadTex(const char * pImageData, int height, int width, int widthStep, int channels) 
{

	int nImageSize = width*height*widthStep;
	int nImageMemSize = width*height*widthStep;

	cudaMallocArray (&cuArray, &imgTex.channelDesc, nImageSize, 1);
	cudaMemcpyToArray(cuArray, 0, 0, pImageData, nImageMemSize, cudaMemcpyHostToDevice);
	// bind a texture to the CUDA array
	cudaBindTextureToArray (imgTex, cuArray);
	// host side settable texture attributes
	imgTex.normalized = false;
	imgTex.filterMode = cudaFilterModeLinear;


}

int main( int argc, char** argv) 
{


	//xLoadImage();

	int nChannels = atoi(argv[3]);
	int nBits = atoi(argv[2]);
	std::string fileName = argv[1];
	GeoTiffRead(fileName.c_str(), nBits,nChannels);
	printf("%d", nWidth);
	runTest();
	GeoTiffShow();
	cutilExit(argc, argv);
}


void runTest( ) 
{

	unsigned int size = nHeight *nWidthStep*sizeof(char);
	unsigned int maskSize = nHeight *nWidth*sizeof(char);
	unsigned int nHistSize = nChannels*2500;
	unsigned int nHistDataSize = nChannels*2500*sizeof(int);
	int *h_Histograms = (int*) malloc(nHistDataSize);
	zeroInitVector(h_Histograms, nHistSize);


	int *h_CDFHistograms = (int*) malloc(nHistDataSize);
	zeroInitVector(h_CDFHistograms, nHistSize);

	cutilSafeCall( cudaThreadSynchronize() );
	unsigned int timer = 0;
	cutilCheckError( cutCreateTimer( &timer));
	cutilCheckError( cutStartTimer( timer));
	// allocate device memory for result

	//cutilSafeCall( cudaMalloc( (void**) &pData, size));

	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
	dim3 dimgTexrid((nWidth+dimBlock.x) / dimBlock.x, (nHeight+dimBlock.y) / dimBlock.y, 1);

	// warmup
	//RGBNIRShadowNDVIKernel<<< dimgTexrid, dimBlock, 0 >>>( d_data,
	//	d_NDVIMaskData,
	//	d_ShadowMaskdata,
	//	//d_Histograms,
	//	nWidth, 
	//	nHeight,
	//	nChannels,
	//	nWidthStep, 
	//	nMaskWidthStep);

	CalculateHistograms( pData,
		h_Histograms,
		nWidth, nHeight,nChannels, nWidthStep, 2500) ;

	StretchImage( pData,
		h_Histograms,
		nWidth, nHeight,nChannels, nWidthStep, 2500) ;

	// execute the kernel
	printf("\nKernel Started!\n");
	int threshold = 0;

	zeroInitVector(h_Histograms, 2048*4);

	CalculateHistograms( pData,
		h_Histograms,
		nWidth, nHeight,nChannels, nWidthStep, 2048) ;

	CalculateCDF( h_Histograms, h_CDFHistograms,
		nWidth, nHeight, nWidth*nHeight,  2048, 4) ;


	HistEqImage(pData,
		h_CDFHistograms,
		nWidth, nHeight,nChannels, nWidthStep, 2048) ;

	NDVIKernel( pData,
		pNDVIMask,
		h_Histograms,
		nWidth, 
		nHeight,
		nChannels,
		nWidthStep, 
		nMaskWidthStep);

	//CalculateHistogram<<< dimgTexrid, dimBlock, 0 >>>( 
	//d_NDVIMaskData,
	//d_Histograms,
	//nWidth, 
	//nHeight,
	//nHeight*nWidth,
	//nMaskWidthStep);



	CalculateThreshold
		( h_Histograms,
		nWidth, 
		nHeight,
		nHeight*nWidth,
		&threshold) ;

	printf("threshold %d\n", threshold);
	cudaThreadSynchronize();
	Threshold
		( pNDVIMask,
		threshold,
		nWidth, 
		nHeight,
		nHeight*nWidth,
		nMaskWidthStep) ;

	//NDVIKernelOtsu<<< dimgTexrid, dimBlock, 0 >>>( d_data,
	//d_NDVIMaskData,
	//d_ShadowMaskdata,
	//d_Histograms,
	//nWidth, 
	//nHeight,
	//nHeight*nWidth,
	//nChannels,
	//nWidthStep, 
	//nMaskWidthStep);

	// check if kernel execution generated an error
	//cutilCheckMsg("Kernel execution failed");



	// allocate mem for the result on host side
	//ValueType * h_odata = (ValueType *) malloc( size);
	// copy result from device to host
	//cutilSafeCall( cudaMemcpy( h_odata, d_data, size, cudaMemcpyDeviceToHost) );
	//cudaMemcpy(pData, d_data, size, cudaMemcpyDeviceToHost);
	//(cudaMemcpy(pData, d_data, size,cudaMemcpyDeviceToHost) );
	//cudaThreadSynchronize();
	//cutilSafeCall( cudaMemcpy( pShadowMask, d_ShadowMaskdata, maskSize, cudaMemcpyDeviceToHost) );
	//cutilSafeCall( cudaMemcpy( pNDVIMask, d_NDVIMaskData, maskSize, cudaMemcpyDeviceToHost) );

	cutilSafeCall( cudaThreadSynchronize() );
	cutilCheckError( cutStopTimer( timer));
	printf("Processing time: %f (ms)\n", cutGetTimerValue( timer));
	printf("%.2f Mpixels/sec\n", (nWidthStep*nHeight / (cutGetTimerValue( timer) / 1000.0f)) / 1e6);
	cutilCheckError( cutDeleteTimer( timer));

	//cutilSafeCall( cudaMemcpy( h_Histograms, d_Histograms, nHistDataSize, cudaMemcpyDeviceToHost) );
	//printMatrix(h_Histograms, 300, 1);
	/*pData=h_odata;*/
	// cleanup memory
	//free(h_odata);
	free(h_Histograms);

	//cutilSafeCall(cudaFree(d_Histograms));
	//cutilSafeCall(cudaFree(d_data));
	//cutilSafeCall(cudaFree(d_ShadowMaskdata));
	//cutilSafeCall(cudaFree(d_NDVIMaskData));
	//cutilSafeCall(cudaFreeArray(cuArray));

	cudaThreadExit();
}