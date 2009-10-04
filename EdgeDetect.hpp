#ifndef TEMPLATE_H_
#define TEMPLATE_H_

#include <CL/cl.h>
#include <string.h>
#include <cstdlib>
#include <iostream>
#include <string>
#include <fstream>

#include <SDKUtil/SDKCommon.hpp>
#include <SDKUtil/SDKApplication.hpp>
#include <SDKUtil/SDKCommandArgs.hpp>
#include <SDKUtil/SDKFile.hpp>

#include <stdio.h>
#include <stdlib.h>

#include </usr/include/opencv/cv.h>
#include </usr/include/opencv/cxcore.h>
#include </usr/include/opencv/highgui.h>

typedef union my_uint4 { cl_uint u32[4]; } my_uint4;
typedef union my_uint2 { cl_uint u32[2]; } my_uint2;

/*** GLOBALS ***/
const bool 		PROFILE 		= false;

const cl_uint ALLOC_TILE = 0;
const cl_uint ALLOC_HORZ = 1;
const cl_uint ALLOC_VERT = 2;

const cl_uint   MASK_WIDTH = 3;        	/**< mask dimensions */
const cl_uint   MASK_HEIGHT = 3;       	/**< mask dimensions */

streamsdk::SDKCommon sampleCommon;

int 		runTimerKey;

cl_double    totalKernelTime;   /**< Time for kernel execution */
cl_double    totalProgramTime;  /**< Time for program execution */
cl_double    referenceKernelTime;/**< Time for reference implementation */

// problem size for 1D algorithm and width of problem size for 2D algorithm
cl_uint width;
cl_uint height;

my_uint4 *input;				// Input data is stored here.
cl_uint  *intermediate;			// Output data is stored here.
cl_uint  *output;				// Output data is stored here.

//// Sobel Operators are stored here.
cl_uint clSobelOpX[9] = {	-1, 0, 1,
							-2, 0, 2,
							-1, 0, 1};
cl_uint clSobelOpY[9] = { 	1, 2, 1,
							0, 0, 0,
							-1,-2,-1};
int cvSobelOpX[3][3] = {{-1, 0, 1},
						{-2, 0, 2},
						{-1, 0, 1}};
int cvSobelOpY[3][3] = {{ 1, 2, 1},
						{ 0, 0, 0},
						{-1,-2,-1}};

// The memory buffer that is used as input/output for OpenCL kernel
cl_mem   inputBuffer;
cl_mem	 intermediateBuffer;
cl_mem	 outputBuffer;
cl_mem	 sobelOpXBuffer;
cl_mem	 sobelOpYBuffer;

cl_context          context;
cl_device_id        *devices;
cl_command_queue    commandQueue;

cl_program program;

/* This program uses only one kernel and this serves as a handle to it */
cl_kernel  kernel;


/*** FUNCTION DECLARATIONS ***/

// Utility funs
void cvDisplay(IplImage* image, char windowName[], int x=0, int y=0);

my_uint4 *cvImageToClArray(IplImage* raw);
IplImage* clArrayToCvImage(cl_uint* output, int width, int height);
void cvMatToCvImage(IplImage* cvImg,CvMat* cvMat);

// OpenCV related funs
void cvGenerateIntensityImage(IplImage* cvImgRaw,CvMat* cvMatIntensity);
void cvGenerateSobelImage(CvMat* cvMatIntensity,CvMat* cvMatSobel);

// OpenCL related funs
void clInitialize(void);
std::string convertToString(const char * filename);

/*
 * This is called once the OpenCL context, memory etc. are set up,
 * the program is loaded into memory and the kernel handles are ready.
 * 
 * It sets the values for kernels' arguments and enqueues calls to the kernels
 * on to the command queue and waits till the calls have finished execution.
 *
 * It also gets kernel start and end time if profiling is enabled.
 */
void clRunKernels(void);

/* Releases OpenCL resources (Context, Memory etc.) */
void clCleanup(void);

/* Releases program's resources */
void clCleanupHost(void);

#endif  /* #ifndef TEMPLATE_H_ */
