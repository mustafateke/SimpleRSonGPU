//============================================================================
// Name        : opencv_gdal.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C, Ansi-style
//============================================================================

#include "opencv/cv.h"
#include "opencv/cxcore.h"
#include "opencv/highgui.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <stddef.h>

typedef char ValueType;
extern int nWidth;
extern int nHeight ;
extern int nChannels;
extern int nWidthStep;
extern int nMaskWidthStep;
extern ValueType * pData ;
extern char * pShadowMask ;
extern char * pNDVIMask ;




#include "gdal_priv.h"

#include "gdal_alg.h"
//#include "ogr_srs_api.h"
#include "cpl_string.h"
#include "cpl_conv.h"
#include "cpl_multiproc.h"


//#define RASTER_COUNT 3
#define DATATYPE GDT_UInt16


using namespace std;

typedef enum fileLoadStatus {
	LOAD_OK = 0,
	LOAD_FAILED = 1
}fileLoadStatus;

/**
* @brief Class that handles loading of raster data.
* Responsible from reading/writing raster data and can be used directly for openCV
* */
class HRaster : public IplImage {
public:
	/** Default constructor */
	HRaster() : IplImage() {
		init();
	}

	/** Destructor */
	virtual ~HRaster() {
		// TODO
		// ds
		// rasterBands
	}

	/** Loads GIS based raster files.
	*  @param fileName Name of the file to be loaded
	* */
	fileLoadStatus load(const char *fileName) {
		ds = (GDALDataset *) GDALOpen(fileName, GA_ReadOnly);
		if (ds==NULL)
			return LOAD_FAILED;



		int RASTER_COUNT = ds->GetRasterCount();
		cout << "NUMBER OF BANDS " << RASTER_COUNT << endl;

		cout << "DESCRIPTION " << ds->GetDriver()->GetDescription() << endl << "META ITEM " << ds->GetDriver()->GetMetadataItem( GDAL_DMD_LONGNAME) << endl;
		for (int i=0;i<RASTER_COUNT;i++) {
			GDALColorInterp temp=ds->GetRasterBand(i+1)->GetColorInterpretation ();
			cout << "BAND TYPE " << temp << endl;
		}
		// TODO the rest of the meta data can also be obtained from ds

		rasterBands = new GDALRasterBand*[RASTER_COUNT];
		for (int i=0;i<RASTER_COUNT;i++)
			rasterBands[i] = NULL;

		for (int i=0;i<RASTER_COUNT;i++) {
			// check the band type so that openCV bgr format can be satisfied
			GDALColorInterp colorInter=ds->GetRasterBand(i+1)->GetColorInterpretation();
			int bandIndex=i;
			switch(colorInter) {
			case GCI_RedBand:
				bandIndex = 2;
				break;
			case GCI_GreenBand:
				bandIndex = 1;
				break;
			case GCI_BlueBand:
				bandIndex = 0;
				break;
			case GCI_AlphaBand:
				bandIndex = 3;
				break;
			default:
				bandIndex = i;
			}
			// TODO support for false color shall be added
			if (bandIndex>=RASTER_COUNT)
				bandIndex=i;
			if (rasterBands[bandIndex]!=NULL) {
				rasterBands[i] = rasterBands[bandIndex];
			}
			rasterBands[bandIndex] = ds->GetRasterBand(i+1);
			//			rasterBands[i] = ds->GetRasterBand(i+1);
		}
		int nXSize = rasterBands[0]->GetXSize();
		int nYSize = rasterBands[0]->GetYSize();
		int nXBlockSize, nYBlockSize;
		rasterBands[0]->GetBlockSize(&nXBlockSize, &nYBlockSize);

		void *blocks=NULL;
		GDALDataType dType = rasterBands[0]->GetRasterDataType();
		// allocate space depending on the raster data type
		int imDepth=IPL_DEPTH_8U; // depth initialized
		size_t stepSize = sizeof(GByte);

		switch (dType) {
		case GDT_Byte:
			blocks = (GByte  *) CPLMalloc(sizeof(GByte ) * nXBlockSize*nYBlockSize);
			imDepth = IPL_DEPTH_8U;
			stepSize = sizeof(GByte);
			break;
		case GDT_UInt16 :
			blocks = (GUInt16   *) CPLMalloc(sizeof(GUInt16  ) * nXBlockSize*nYBlockSize);
			imDepth = IPL_DEPTH_16U;
			stepSize = sizeof(GUInt16);
			break;
		case GDT_Int16  :
			blocks = (GInt16    *) CPLMalloc(sizeof(GInt16   ) * nXBlockSize*nYBlockSize);
			imDepth = IPL_DEPTH_16S;
			stepSize = sizeof(GInt16);
			break;
		case GDT_UInt32  :
			blocks = (GUInt32    *) CPLMalloc(sizeof(GUInt32   ) * nXBlockSize*nYBlockSize);
			imDepth = IPL_DEPTH_32F; // TODO : This is unclear. There is no IPL_DEPTH_32U
			stepSize = sizeof(GUInt32);
			break;
		case GDT_Int32  :
			blocks = (GInt32    *) CPLMalloc(sizeof(GInt32   ) * nXBlockSize*nYBlockSize);
			imDepth = IPL_DEPTH_32S;
			stepSize = sizeof(GInt32);
			break;
		case GDT_Float32  :
			blocks = (float     *) CPLMalloc(sizeof(float    ) * nXBlockSize*nYBlockSize);
			imDepth = IPL_DEPTH_32F;
			stepSize = sizeof(float);
			break;
		case GDT_Float64   :
			blocks = (double      *) CPLMalloc(sizeof(double     ) * nXBlockSize*nYBlockSize);
			imDepth = IPL_DEPTH_64F;
			stepSize = sizeof(double);
			break;
		case GDT_CInt16   :
			// TODO what if the image is complex. stepSize needs to be taken care as well.
			blocks = (GInt16      *) CPLMalloc(sizeof(GInt16     ) * nXBlockSize*nYBlockSize*2);
			break;
		case GDT_CInt32   :
			blocks = (GInt32      *) CPLMalloc(sizeof(GInt32     ) * nXBlockSize*nYBlockSize*2);
			break;
		case GDT_CFloat32    :
			blocks = (float       *) CPLMalloc(sizeof(float      ) * nXBlockSize*nYBlockSize*2);
			break;
		case GDT_CFloat64     :
			blocks = (double        *) CPLMalloc(sizeof(double       ) * nXBlockSize*nYBlockSize*2);
			break;
		default:
			cout << "ERROR: UNKNOWN DATA TYPE" << endl;
			break;
		}

		int nXBlocks = (nXSize-1) / nXBlockSize + 1;
		int nYBlocks = (nYSize-1) / nYBlockSize + 1;

		// create IplImage header and allocate underlying data
		cvInitImageHeader( (IplImage*)this, cvSize(nXSize, nYSize), imDepth, RASTER_COUNT, IPL_ORIGIN_TL,CV_DEFAULT_IMAGE_ROW_ALIGN);
		cout << "HEADER CREATED FOR SIZE " << nXSize << "X" << nYSize << endl;
		cvCreateData((IplImage*)this);

		void * geoData = this->imageData;
		int step = this->widthStep / stepSize;

		for(int bandIndex = 0; bandIndex<RASTER_COUNT; bandIndex++){
			int index, nXValid, nYValid;
			for(int iYBlock = 0; iYBlock<nYBlocks; iYBlock++){
				for(int iXBlock = 0; iXBlock<nXBlocks; iXBlock++){
					rasterBands[bandIndex]->ReadBlock(iXBlock, iYBlock, blocks);;

					// Compute the portion of the block that is valid
					// for partial edge blocks.
					if( (iXBlock+1) * nXBlockSize > nXSize )
						nXValid = nXSize - iXBlock * nXBlockSize;
					else
						nXValid = nXBlockSize;

					if( (iYBlock+1) * nYBlockSize > nYSize )
						nYValid = nYSize - iYBlock * nYBlockSize;
					else
						nYValid = nYBlockSize;

					for( int iY = 0; iY < nYValid; iY++ ){
						for( int iX = 0; iX < nXValid; iX++ ){
							index = (iYBlock*nYBlockSize + iY) * step + (iXBlock*nXBlockSize + iX) * RASTER_COUNT + bandIndex;
							// we need to access the geoData and blocks. So they need to be casted
							switch (dType) {
								case GDT_Byte:
									((GByte  *)geoData)[index] = ((GByte *)blocks)[iX + iY*nXBlockSize];
									break;
								case GDT_UInt16 :
									((GUInt16  *)geoData)[index] = ((GUInt16 *)blocks)[iX + iY*nXBlockSize];
									break;
								case GDT_Int16  :
									((GInt16  *)geoData)[index] = ((GInt16 *)blocks)[iX + iY*nXBlockSize];
									break;
								case GDT_UInt32  :
									((GUInt32  *)geoData)[index] = ((GUInt32 *)blocks)[iX + iY*nXBlockSize];
									break;
								case GDT_Int32  :
									((GInt32  *)geoData)[index] = ((GInt32 *)blocks)[iX + iY*nXBlockSize];
									break;
								case GDT_Float32  :
									((float  *)geoData)[index] = ((float *)blocks)[iX + iY*nXBlockSize];
									break;
								case GDT_Float64   :
									((double  *)geoData)[index] = ((double *)blocks)[iX + iY*nXBlockSize];
									break;
								case GDT_CInt16   :
								case GDT_CInt32   :
								case GDT_CFloat32    :
								case GDT_CFloat64     :
								default:
									cout << "ERROR: UNKNOWN DATA TYPE" << endl;
									break;
							}
						}
					}
				}
			}
		}

		imageLoaded = true;
		// TODO
		// blocks
		return LOAD_OK;
	}

	void convertTo8BitGray(void) {
		if (!imageLoaded) {
			cout << "ERROR: NO IMAGE HAS BEEN LOADED!!" << endl;
		}
		if (grey8Bit!=NULL && grey8Bit!=((IplImage *)this)) {
			cvReleaseImage(&grey8Bit);
		}
		// use the first 3 channels to get the gray scale
		// TODO working for 3 and 4 bands. Generalize it by using one band per channel
		//	if (this->nChannels==1) { // data type can be different
		//		grey8Bit=(IplImage *)this;
		//		return;
		//	}
		grey8Bit=cvCreateImage(cvGetSize(((IplImage *)this)), IPL_DEPTH_8U, 1);
		cout << "IMAGE CREATED" << endl;
		cvConvertScale((IplImage *)this, grey8Bit, 1.0,0.0);
		//	cvCvtColor((IplImage *)this, grey8Bit, CV_BGR2GRAY);
		cout << "GRAY SCALE" << endl;
	}

	void convertTo24BitGray(int dataSize = 8) {
		if (!imageLoaded) {
			cout << "ERROR: NO IMAGE HAS BEEN LOADED!!" << endl;
		}
		if (grey8Bit!=NULL && grey8Bit!=((IplImage *)this)) {
			cvReleaseImage(&grey8Bit);
		}
		if (Blue!=NULL) {
			cvReleaseImage(&Blue);
		}
		if (Red!=NULL ) {
			cvReleaseImage(&Red);
		}
		if (Green!=NULL) {
			cvReleaseImage(&Green);
		}
		if (NIR!=NULL) {
			cvReleaseImage(&NIR);
		}
		if (RGB!=NULL) {
			cvReleaseImage(&RGB);
		}
		Blue = cvCreateImage(cvGetSize(((IplImage *)this)), depth, 1);
		Blue8 = cvCreateImage(cvGetSize(((IplImage *)this)), IPL_DEPTH_8U, 1);
		Red = cvCreateImage(cvGetSize(((IplImage *)this)), depth, 1);
		Red8 = cvCreateImage(cvGetSize(((IplImage *)this)), IPL_DEPTH_8U, 1);
		Green = cvCreateImage(cvGetSize(((IplImage *)this)), depth, 1);
		Green8 = cvCreateImage(cvGetSize(((IplImage *)this)), IPL_DEPTH_8U, 1);
		NIR = cvCreateImage(cvGetSize(((IplImage *)this)), depth, 1);
		NIR8 = cvCreateImage(cvGetSize(((IplImage *)this)), IPL_DEPTH_8U, 1);
		RGB = cvCreateImage(cvGetSize(((IplImage *)this)), IPL_DEPTH_8U, 3);
		RGBNIR = cvCreateImage(cvGetSize(((IplImage *)this)), IPL_DEPTH_16U, nChannels);
		RGBNIR8 = cvCreateImage(cvGetSize(((IplImage *)this)), IPL_DEPTH_8U, nChannels);
		HSV = cvCreateImage(cvGetSize(((IplImage *)this)), IPL_DEPTH_8U, 3);

		IplImage * RGBTemp = cvCreateImage(cvGetSize(((IplImage *)this)), depth, 3);

		ShadowMask = cvCreateImage(cvGetSize(((IplImage *)this)), IPL_DEPTH_8U, 1);
		NDVIMask = cvCreateImage(cvGetSize(((IplImage *)this)), IPL_DEPTH_8U, 1);

		cvSplit((IplImage *)this, Red, Green, Blue, NIR);
		cvConvertScale(Blue, Blue8, 255/2300.0,0.0);
		cvConvertScale(Red, Red8, 255/2300.0,0.0);
		cvConvertScale(Green, Green8, 255/2300.0,0.0);
		cvConvertScale(NIR, NIR8, 255/2300.0,0.0);
		cvMerge(Blue, Green, Red,0, RGBTemp);
		/*cvMerge( Red8, Green8, Blue8,NIR8, RGBNIR8);*/
		//Switch R and B
		cvMerge( Blue8, Green8, Red8,NIR8, RGBNIR8);

		// use the first 3 channels to get the gray scale
		// TODO working for 3 and 4 bands. Generalize it by using one band per channel
		//	if (this->nChannels==1) { // data type can be different
		//		grey8Bit=(IplImage *)this;
		//		return;
		//	}
		//grey8Bit=cvCreateImage(cvGetSize(((IplImage *)this)), IPL_DEPTH_8U, 1);
		cout << "IMAGE CREATED" << endl;
		//cvConvertScale((IplImage *)this, grey8Bit, 1.0,0.0);

		cvConvertScale(RGBTemp, RGB, (pow((double)2,(int)8)-1)/2047.0,0.0);
		//RGBNIR=(IplImage *)this;
		cvConvertScale((IplImage *)this, RGBNIR, 1/*(pow((double)2,(int)8)-1)/2047.0*/,0.0);
		double minVal;
		double maxVal;
		cvMinMaxLoc(Red,&minVal , &maxVal);
		
		cvCvtColor(RGB, HSV, CV_RGB2HSV); 

		//	cvCvtColor((IplImage *)this, grey8Bit, CV_BGR2GRAY);
		cout << "GRAY SCALE" << endl;
		if (RGBTemp!=NULL) {
			cvReleaseImage(&RGBTemp);
		}
	}

	void save8BitGrey(const char *fileName) {
		cvSaveImage(fileName, grey8Bit);
	}

private:


protected:
	/** Intializes the internal attributes */
	void init() {
		ds = NULL;
		grey8Bit = NULL;
		// TODO add a reset functionality
		imageLoaded = false;
		grey8Bit = NULL;
		Blue = NULL;
		Red = NULL;
		Green = NULL;
		NIR = NULL;
		RGB = NULL;
	}

protected:
	/// Internal raster representation adopted from GDAL library
	GDALDataset *ds;

	/// Specifies whether an image is loaded or not
	bool imageLoaded;

	/// Pointer array to the raster bands
	GDALRasterBand  **rasterBands;

public:
	/// 8 bit gray-scale version of the image. Essential for most opencv operations
	IplImage *grey8Bit;
	IplImage *Blue;
	IplImage *Blue8;
	IplImage *Red;
	IplImage *Red8;
	IplImage *Green;
	IplImage *Green8;
	IplImage *NIR;
	IplImage *NIR8;
	IplImage *RGB;
	IplImage *RGBNIR;
	IplImage *RGBNIR8;
	IplImage *ShadowMask;
	IplImage *NDVIMask;
	IplImage *HSV;


	IplImage *GetIplImage()
	{
		return (IplImage*)this;
	}
};
HRaster mL;
int GeoTiffRead(const char *fileName, int dataSize = 8, int channels = 3) {

	GDALAllRegister();

	fileLoadStatus a = mL.load(fileName);

	cout << "IMAGE LOADED" << endl;
	mL.convertTo24BitGray(16);

	//mL.save8BitGrey("lgray.png");
	cout << "CONVERSION OK" << endl;

	pShadowMask = mL.ShadowMask->imageData;
	pNDVIMask = mL.NDVIMask->imageData;
	nMaskWidthStep=mL.ShadowMask->widthStep;

	if(channels == 3)
	{
		nWidth = mL.RGB->width;
		nHeight = mL.RGB->height;
		nChannels = mL.RGB->nChannels;
		nWidthStep = mL.RGB->widthStep;
		//pData = mL.RGB->imageData;	
		pData = (ValueType *)mL.HSV->imageData;	
		nWidth=mL.RGB->width;
		//cvNamedWindow("test");
		//cvShowImage("test", mL.RGB);
		//cvWaitKey(-1);
	}
	else if(channels == 4)
	{
		nWidth = mL.RGBNIR->width;
		nHeight = mL.RGBNIR->height;
		nChannels = mL.RGBNIR->nChannels;
		nWidthStep = mL.RGBNIR->widthStep;
		pData = (ValueType *)mL.RGBNIR->imageData;	
		//pData = mL.HSV->imageData;	
		nWidth=mL.RGBNIR->width;
		//cvNamedWindow("test");
		//cvShowImage("test", mL.GetIplImage());
		//cvWaitKey(-1);
	}

	return 0;
}

int GeoTiffShow(void) {

	GDALAllRegister();


	mL.RGB->imageData=(char *)pData;
	/*mL.convertTo24BitGray();*/

	mL.ShadowMask->imageData=pShadowMask;
	mL.NDVIMask->imageData=pNDVIMask;

	cvNamedWindow("test");
	cvShowImage("test", mL.NDVIMask);
	cvSaveImage("ndvi.jpg", mL.NDVIMask);
	cvSaveImage("shadow.jpg", mL.ShadowMask);
	cvWaitKey(-1);

	return 0;
}

//int main(void) {
///*
//	IplImage* srcLeft = cvLoadImage("Left.jpg",1);
//
//	IplImage* srcRight = cvLoadImage("Right.jpg",1);
//
//	IplImage* leftImage = cvCreateImage(cvGetSize(srcLeft), IPL_DEPTH_8U, 1);
//
//	IplImage* rightImage = cvCreateImage(cvGetSize(srcRight), IPL_DEPTH_8U, 1);
//
//	cvCvtColor(srcLeft, leftImage, CV_BGR2GRAY);
//
//	cvCvtColor(srcRight, rightImage, CV_BGR2GRAY);
//
//	CvSize size = cvGetSize(srcLeft);
//
//	CvMat* disparity_left = cvCreateMat( size.height, size.width, CV_16S );
//
//	CvMat* disparity_right = cvCreateMat( size.height, size.width, CV_16S );
//
//	CvStereoGCState* state = cvCreateStereoGCState( 16, 2 );
//
//	cvFindStereoCorrespondenceGC( leftImage, rightImage,disparity_left, disparity_right, state, 0 );
//
//	cvReleaseStereoGCState( &state );
//
//	CvMat* disparity_left_visual = cvCreateMat( size.height, size.width, CV_8U );
//
//	cvConvertScale( disparity_left, disparity_left_visual, -16 );
//	cvSaveImage("disparity.jpg", disparity_left_visual);
//*/
//
//
//	GDALAllRegister();
//	HRaster mL;
//	fileLoadStatus a = mL.load("C:\\emre\\veri\\frame\\col90p1.img");
////	fileLoadStatus a = mL.load("C:\\emre\\veri\\laguna_beach\\lag11p1.img");
////	fileLoadStatus a = mL.load("Left.jpg");
////	fileLoadStatus a = mL.load("left1.png");
////	fileLoadStatus a = mL.load("left2.png");
////	fileLoadStatus a = mL.load("lgrayO.png");
//	cout << "IMAGE LOADED" << endl;
//	mL.convertTo8BitGray();
//	mL.save8BitGrey("lgray.png");
//	cout << "CONVERSION OK" << endl;
////	mL.save8BitGrey("deneme2.png");
//
//	HRaster mR;
//	a = mR.load("C:\\emre\\veri\\frame\\col91p1.img");
////	a = mR.load("C:\\emre\\veri\\laguna_beach\\lag12p1.img");
////	a = mR.load("Right.jpg");
////	a = mR.load("right1.png");
////	a = mR.load("right2.png");
////	a = mR.load("rgrayO.png");
//	mR.convertTo8BitGray();
//	mR.save8BitGrey("rgray.png");
//
//
///*
//	HRaster mL,mR;
//	IplImage* srcLeft = cvLoadImage("Left.jpg",1);
//	IplImage* srcRight = cvLoadImage("Right.jpg",1);
//
//	mL.grey8Bit = cvCreateImage(cvGetSize(srcLeft), IPL_DEPTH_8U, 1);
//
//	mR.grey8Bit = cvCreateImage(cvGetSize(srcRight), IPL_DEPTH_8U, 1);
//
//	cvCvtColor(srcLeft, mL.grey8Bit, CV_BGR2GRAY);
//
//	cvCvtColor(srcRight, mR.grey8Bit, CV_BGR2GRAY);
//*/
//	cout << "PREPARED FOR DEPTH PREDICTION" << endl;
//	// image_left and image_right are the input 8-bit single-channel images
//	// from the left and the right cameras, respectively
//	CvSize size = cvGetSize(mL.grey8Bit);
//	cout << size.height << " " << size.width << endl;
//	CvMat* disparity_left = cvCreateMat( size.height, size.width, CV_16S );
//	CvMat* disparity_right = cvCreateMat( size.height, size.width, CV_16S );
////	CvStereoGCState* state = cvCreateStereoGCState( 16, 2 );
//
//
///*
//	CvStereoGCState* state = cvCreateStereoGCState( 16, 1 );
//	cout << "disparities " << state->minDisparity << " " << state->numberOfDisparities << endl;
//	state->minDisparity = 130;
//	state->numberOfDisparities = 10;
//	cout << "FINDING CORRESPONDANCES" << endl;
//	cvFindStereoCorrespondenceGC( mL.grey8Bit, mR.grey8Bit,disparity_left, disparity_right, state, 0 );
//	cvReleaseStereoGCState( &state );
//*/
//
//	CvStereoBMState* state = cvCreateStereoBMState(2);
//	state->preFilterSize = 5;
//	state->preFilterCap = 31;
//	state->SADWindowSize = 15;
//	state->minDisparity = 0;
//	state->numberOfDisparities = 1024;//256;
//	state->textureThreshold = 3;
//	state->uniquenessRatio = 1;
//	state->speckleWindowSize = 100;
////	state->speckleRange = 4;
////	state->disp12MaxDiff = 1;
//	cvFindStereoCorrespondenceBM(mL.grey8Bit, mR.grey8Bit,disparity_left,state);
//	state->disp;
//	cvReleaseStereoBMState(&state);
//
//
//	cout << "STATE RELEASED" << endl;
//	// now process the computed disparity images as you want
//	CvMat* disparity_left_visual = cvCreateMat( size.height, size.width, CV_8U );
////	cvConvertScale( disparity_left, disparity_left_visual, -16 );
//	cvConvertScale( disparity_left, disparity_left_visual );
//	CvMat *vDisp = cvCreateMat(size.height, size.width, CV_8U );
//	cvNormalize(disparity_left,vDisp,0,256,CV_MINMAX);
//	cvSaveImage( "disparity.jpg", vDisp);
//
//
//
//
//
////	fileLoadStatus a = mR.load("5m.tif");
////	fileLoadStatus a = mR.load("shadow.jpg");
////	cout << "file load status " << a << endl;
//
//	return 0;
//}
