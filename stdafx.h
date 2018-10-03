// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#pragma once
// Thread block size
#define BLOCK_SIZE 32
#define WARP_SIZE 32
#define GRAYLEVEL 256
void zeroInitVector(int* data, int size)
{
	for (int i = 0; i < size; ++i)
		data[i] = 0; //rand()%10 /*/ (int)RAND_MAX*/;
}

void printMatrix(int *Matrix, int heigth, int width, char* name = "")
{
	//printf("\n %s  Mat %d x %d\n",name, heigth, width);
	//for (int rowIndex = 0; rowIndex < heigth; rowIndex++)
	//	for (int colIndex = 0; colIndex < width; colIndex++)
	//	{

	//		printf("%d\t", Matrix[rowIndex * (width) + colIndex]);
	//		if(colIndex == width-1)
	//		{
	//			printf("\n");
	//		}
	//	}
}
//void * LoadImage();

// TODO: reference additional headers your program requires here
