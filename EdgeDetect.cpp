#include "EdgeDetect.hpp"
#include <malloc.h>

void cvDisplay(IplImage* image, char windowName[], int x, int y)
{
	CvSize imageSize = cvGetSize(image);
	cvNamedWindow(windowName);
	cvResizeWindow(windowName, imageSize.width, imageSize.height);
	cvMoveWindow(windowName, x, y);
	cvShowImage(windowName,image);
	cvWaitKey(0);
	cvDestroyAllWindows();
}

my_uint4 *cvImageToClArray(IplImage* raw)
{
	int width = raw->width;
	int height = raw->height;

	my_uint4 *imageArray = (my_uint4*)malloc(width * height * sizeof(my_uint4));

	for (int y=0; y<height; y++)
	{
		for (int x=0; x<width; x++)
		{
			CvScalar colourValue = cvGet2D(raw,y,x);

			int index = (y*width) + x;

			imageArray[index].u32[0] = (cl_uint)(colourValue.val[0]); 	//B
			imageArray[index].u32[1] = (cl_uint)(colourValue.val[1]); 	//G
			imageArray[index].u32[2] = (cl_uint)(colourValue.val[2]); 	//R
			imageArray[index].u32[3] = (cl_uint)(0); 					//A
		}
	}

	return imageArray;
}

//converts raw image into intensity values
IplImage* clArrayToCvImage(cl_uint* output, int resultWidth, int resultHeight)
{
	CvSize size;
	size.width  = resultWidth;
	size.height = resultHeight;

	IplImage* resultImg = cvCreateImage(size,IPL_DEPTH_8U,1);

	//generate intensity image
	for (int y=0; y<resultHeight; y++)
	{
		for (int x=0; x<resultWidth; x++)
		{
			CvScalar colourSelect;

			int index = (y*resultWidth) + x;

			colourSelect.val[0] = output[index];

			cvSet2D(resultImg,y,x,colourSelect);
		}
	}

	return resultImg;
}

void cvMatToCvImage(IplImage* cvImg,CvMat* cvMat)
{
	for (int y=0; y<cvMat->height; y++)
		for (int x=0; x<cvMat->width; x++)
		{
			CvScalar colourSelect;
			colourSelect.val[0] = cvmGet(cvMat,y,x);
			cvSet2D(cvImg,y,x,colourSelect);
		}
}

// Converts the contents of a file into a string
std::string convertToString(const char *filename)
{
	size_t size;
	char*  str;
	std::string s;

	std::fstream f(filename, (std::fstream::in | std::fstream::binary));

	if(f.is_open())
	{
		size_t fileSize;
		f.seekg(0, std::fstream::end);
		size = fileSize = f.tellg();
		f.seekg(0, std::fstream::beg);

		str = new char[size+1];
		if(!str)
		{
			f.close();
			return NULL;
		}

		f.read(str, fileSize);
		f.close();
		str[size] = '\0';

		s = str;

		return s;
	}
	return NULL;
}

//converts raw image into intensity values
void cvGenerateIntensityImage(my_uint4* clArrRaw,cl_uint* clArrIntensity, int height, int width)
{
	//generate intensity image
	for (int y=0; y<height; y++)
		for (int x=0; x<width; x++)
		{
			int index = (y*width) + x;
			clArrIntensity[index] = (clArrRaw[index].u32[0]+clArrRaw[index].u32[1]+clArrRaw[index].u32[2])/3;
		}
}


void cvGenerateSobelImage(cl_uint* clArrIntensity,cl_uint* clArrSobel, int height, int width)
{
	cl_uint Gx;
	cl_uint Gy;

	//generate sobel image
	for (int y=1; y<height-1; y++)
		for (int x=1; x<width-1; x++)
		{
			int index = (y * width) + x;

			Gx = 	(clArrIntensity[index - width - 1] 	*   clSobelOpX[0]) +
					(clArrIntensity[index - width] 		*   clSobelOpX[1]) +
					(clArrIntensity[index - width + 1]	*   clSobelOpX[2]) +

					(clArrIntensity[index - 1]			*   clSobelOpX[3]) +
					(clArrIntensity[index]				*   clSobelOpX[4]) +
					(clArrIntensity[index + 1]			*   clSobelOpX[5]) +

					(clArrIntensity[index + width - 1]	*   clSobelOpX[6]) +
					(clArrIntensity[index + width]		*   clSobelOpX[7]) +
					(clArrIntensity[index + width + 1]	*   clSobelOpX[8]);
			Gx =	abs(Gx);

			Gy = 	(clArrIntensity[index - width - 1] 	*  	clSobelOpY[0]) +
					(clArrIntensity[index - width] 		*   clSobelOpY[1]) +
					(clArrIntensity[index - width + 1]	*   clSobelOpY[2]) +

					(clArrIntensity[index - 1]			*   clSobelOpY[3]) +
					(clArrIntensity[index]				*   clSobelOpY[4]) +
					(clArrIntensity[index + 1]			*   clSobelOpY[5]) +

					(clArrIntensity[index + width - 1]	*   clSobelOpY[6]) +
					(clArrIntensity[index + width]		*   clSobelOpY[7]) +
					(clArrIntensity[index + width + 1]	*   clSobelOpY[8]);
			Gy =	abs(Gy);

			clArrSobel[index] = Gx+Gy;
		}
}

/////////////////////////////////////////////////////////////////
// Parallel (OpenCL) Methods
/////////////////////////////////////////////////////////////////

// Host Initialization: Allocate & init memory on the host. Print input array.
void clInitializeHost(IplImage* cvRawImg)
{
    if(input != NULL)
    {
        free(input);
        input = NULL;
    }

	if(intermediate != NULL)
    {
        free(intermediate);
        intermediate = NULL;
    }

    if(output != NULL)
    {
        free(output);
        output = NULL;
    }

	input = cvImageToClArray(cvRawImg);
    if(input==NULL)
    {
    	std::cout<<"Error: Failed to allocate host memory. (input)\n";
        return;
    }

    intermediate = (cl_uint*)malloc(width * height * sizeof(cl_uint));
    if(intermediate==NULL)
    {
    	std::cout<<"Error: Failed to allocate host memory. (intermediate)\n";
        return;
    }

    output = (cl_uint*)malloc(width * height * sizeof(cl_uint));
    if(output==NULL)
    {
    	std::cout<<"Error: Failed to allocate host memory. (output)\n";
        return;
    }
}

// OpenCL related initialization
// -> Create Context, Device list, Command Queue
// -> Create OpenCL memory buffer objects
// -> Load CL file, compile, link CL source
// -> Build program and kernel objects
void clInitialize(void)
{
    cl_int status = 0;
    size_t deviceListSize;

	/////////////////////////////////////////////////////////////////
	// Create an OpenCL context
	/////////////////////////////////////////////////////////////////

    //todo: experiment with CL_DEVICE_TYPE_ ALL, DEFAULT, GPU, ACCELERATOR, CPU
    context = clCreateContextFromType(0, 
                                      CL_DEVICE_TYPE_CPU,
                                      NULL, 
                                      NULL, 
                                      &status);

    if(status != CL_SUCCESS) 
	{  
		std::cout<<"Error: Creating Context. (clCreateContextFromType)\n";
		return; 
	}

    /* First, get the size of device list data */
    status = clGetContextInfo(context, 
                              CL_CONTEXT_DEVICES, 
                              0, 
                              NULL, 
                              &deviceListSize);
    if(status != CL_SUCCESS) 
	{  
		std::cout<< "Error: Getting Context Info (device list size, clGetContextInfo)\n";
		return;
	}

	/////////////////////////////////////////////////////////////////
	// Detect OpenCL devices
	/////////////////////////////////////////////////////////////////
    devices = (cl_device_id *)malloc(deviceListSize);
	if(devices == 0)
	{
		std::cout<<"Error: No devices found.\n";
		return;
	}

    /* Now, get the device list data */
    status = clGetContextInfo(
			     context, 
                 CL_CONTEXT_DEVICES, 
                 deviceListSize, 
                 devices, 
                 NULL);
    if(status != CL_SUCCESS) 
	{ 
		std::cout<<"Error: Getting Context Info (device list, clGetContextInfo)\n";
		return;
	}

	/////////////////////////////////////////////////////////////////
	// Create an OpenCL command queue
	/////////////////////////////////////////////////////////////////
    /* The block is to move the declaration of prop closer to its use */
   	cl_command_queue_properties prop = 0;

   	if (PROFILE)
   	{
   		prop |= CL_QUEUE_PROFILING_ENABLE;
   	}

    commandQueue = clCreateCommandQueue(
    		context,
            devices[0],
            prop,
            &status);
    if(status != CL_SUCCESS)
   	{
   		std::cout<<"Creating Command Queue. (clCreateCommandQueue)\n";
   		return;
   	}

	/////////////////////////////////////////////////////////////////
	// Create OpenCL memory buffers
	/////////////////////////////////////////////////////////////////
    inputBuffer = clCreateBuffer(
            context,
            CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
            sizeof(cl_uint4) * width * height,
            input,
            &status);
    if(status == CL_INVALID_CONTEXT)
    {
		//context not valid
		std::cout<<"Error: clCreateBuffer - invalid context - (inputBuffer)\n";
		return;
    }
    if(status == CL_INVALID_VALUE)
    {
    	//flags value not valid
    	std::cout<<"Error: clCreateBuffer - invalid flags value - (inputBuffer)\n";
		return;
    }
    if(status == CL_INVALID_BUFFER_SIZE)
    {
    	//size==0 or size>CL_DEVICE_MAX_MEM_ALLOC_SIZE
		std::cout<<"Error: clCreateBuffer - invalid buffer size - (inputBuffer)\n";
		return;
    }
    if(status == CL_INVALID_HOST_PTR)
    {
        //(host_ptr == NULL) && (CL_MEM_USE_HOST_PTR || CL_MEM_COPY_HOST_PTR in flags)
    	//||
    	//(host_ptr != NULL) && (CL_MEM_COPY_HOST_PTR || CL_MEM_USE_HOST_PTR _not_ in flags)
    	bool isNull = (input==NULL);
		std::cout<<"Error: clCreateBuffer - invalid host pointer - (inputBuffer) - NULL==" << isNull << "\n";
		return;
    }
    if(status == CL_MEM_OBJECT_ALLOCATION_FAILURE)
    {
        //there is a failure to allocate memory for buffer object
		std::cout<<"Error: clCreateBuffer - mem object alloc failure - (inputBuffer)\n";
		return;
    }
    if(status == CL_OUT_OF_HOST_MEMORY)
    {
        //there is a failure to allocate resources required by the OpenCL implementation on the host
		std::cout<<"Error: clCreateBuffer - out of host mem - (inputBuffer)\n";
		return;
    }
    if(status != CL_SUCCESS) 
	{ 
		std::cout<<"Error: clCreateBuffer (inputBuffer)\n";
		return;
	}

    intermediateBuffer = clCreateBuffer(
							context,
							CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
							sizeof(cl_uint) * width * height,
							intermediate,
							&status);
    if(status != CL_SUCCESS)
	{
		std::cout<<"Error: clCreateBuffer (intermediateBuffer)\n";
		return;
	}

    outputBuffer = clCreateBuffer(
							context,
							CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
							sizeof(cl_uint) * width * height,
							output,
							&status);
    if(status == CL_INVALID_CONTEXT)
    {
		//context not valid
		std::cout<<"Error: clCreateBuffer - invalid context\n";
    }
    if(status == CL_INVALID_VALUE)
    {
    	//flags value not valid
    	std::cout<<"Error: clCreateBuffer - invalid flags value\n";
    }
    if(status == CL_INVALID_BUFFER_SIZE)
    {
    	//size==0 or size>CL_DEVICE_MAX_MEM_ALLOC_SIZE
		std::cout<<"Error: clCreateBuffer - invalid buffer size\n";
    }
    if(status == CL_INVALID_HOST_PTR)
    {
        //(host_ptr == NULL) && (CL_MEM_USE_HOST_PTR || CL_MEM_COPY_HOST_PTR in flags)
    	//||
    	//(host_ptr != NULL) && (CL_MEM_COPY_HOST_PTR || CL_MEM_USE_HOST_PTR _not_ in flags)
    	bool isNull = (input==NULL);
		std::cout<<"Error: clCreateBuffer - invalid host pointer - NULL==" << isNull << "\n";
    }
    if(status == CL_MEM_OBJECT_ALLOCATION_FAILURE)
    {
        //there is a failure to allocate memory for buffer object
		std::cout<<"Error: clCreateBuffer - mem object alloc failure\n";
    }
    if(status == CL_OUT_OF_HOST_MEMORY)
    {
        //there is a failure to allocate resources required by the OpenCL implementation on the host
		std::cout<<"Error: clCreateBuffer - out of host mem\n";
    }
    if(status != CL_SUCCESS)
	{
		std::cout<<"Error: clCreateBuffer (outputBuffer)\n";
		return;
	}

    sobelOpXBuffer = clCreateBuffer(
            context,
            CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
            sizeof(cl_uint ) * MASK_HEIGHT * MASK_WIDTH,
            clSobelOpX,
            &status);
    if(status != CL_SUCCESS)
	{
		std::cout<<"Error: clCreateBuffer (sobelOpXBuffer)\n";
		return;
	}

    sobelOpYBuffer = clCreateBuffer(
            context,
            CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
            sizeof(cl_uint ) * MASK_HEIGHT * MASK_WIDTH,
            clSobelOpY,
            &status);
    if(status != CL_SUCCESS)
	{
		std::cout<<"Error: clCreateBuffer (sobelOpYBuffer)\n";
		return;
	}

	/////////////////////////////////////////////////////////////////
	// Load CL file, build CL program object, create CL kernel object
	/////////////////////////////////////////////////////////////////
    const char * filename  = "EdgeDetect_Kernels.cl";
    std::string  sourceStr = convertToString(filename);
    const char * source    = sourceStr.c_str();
    size_t sourceSize[]    = { strlen(source) };

    //std::cout << source << "\n";

    program = clCreateProgramWithSource(
			      context, 
                  1, 
                  &source,
				  sourceSize,
                  &status);
	if(status != CL_SUCCESS) 
	{ 
	  std::cout<<"Error: Loading Binary into cl_program (clCreateProgramWithSource)\n";
	  return;
	}

    /* create a cl program executable for all the devices specified */
    status = clBuildProgram(program, 1, devices, NULL, NULL, NULL);

    //error checking code

    if(!sampleCommon.checkVal(status,CL_SUCCESS,"clBuildProgram failed."))
    {
       //print kernel compilation error
       char programLog[4096];

       cl_int tempStatus = clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, 4096, programLog, 0);
       std::cout<<"\n---Build Log---\n"<<programLog<<"\n---Build Log---\n";
       return;
    }

    if(status == CL_INVALID_PROGRAM)
   	{
    	//if program is not a valid program object.
   		std::cout<<"Error: Invalid program object. (clBuildProgram)\n";
   		return;
   	}
    if(status == CL_INVALID_VALUE)
   	{
        // (device_list == NULL) && (num_devices > 0)
    	// ||
        // (device_list != NULL) && (num_devices ==0)
    	// ||
        // (pfn_notify == NULL) && (user_data != NULL)
   		std::cout<<"Error: Invalid value - device_list==NULL:" << (devices==NULL) << " - (clBuildProgram)\n";
   		return;
   	}
    if(status == CL_INVALID_DEVICE)
   	{
        // OpenCL devices listed in device_list are not in the list of
        // devices associated with program.
   		std::cout<<"Error: Invalid device. (clBuildProgram)\n";
   		return;
   	}
    if(status == CL_INVALID_BINARY)
   	{
        // if program is created with clCreateWithProgramBinary and
        // devices listed in device_list do not have a valid program binary loaded.
   		std::cout<<"Error: Invalid binary. (clBuildProgram)\n";
   		return;
   	}
    if(status == CL_INVALID_BUILD_OPTIONS)
   	{
        // if the build options specified by options are invalid
   		std::cout<<"Error: Invalid build options. (clBuildProgram)\n";
   		return;
   	}
    if(status == CL_INVALID_OPERATION)
   	{
        // if the build of a program executable for any of the devices
        // listed in device_list by a previous call to clBuildProgram for program has not
        // completed
    	// ||
    	// if there are kernel objects attached to program.
   		std::cout<<"Error: Invalid operation. (clBuildProgram)\n";
   		return;
   	}
    if(status == CL_COMPILER_NOT_AVAILABLE)
   	{
        // CL_COMPILER_NOT_AVAILABLE if program is created with
        // clCreateProgramWithSource and a compiler is not available i.e.
        // CL_DEVICE_COMPILER_AVAILABLE specified in table 4.3 is set to CL_FALSE.
   		std::cout<<"Error: Compiler not available. (clBuildProgram)\n";
   		return;
   	}
    if(status == CL_BUILD_PROGRAM_FAILURE)
   	{
        // if there is a failure to build the program executable.
        // This error will be returned if clBuildProgram does not return until the build has
        // completed.
   		std::cout<<"Error: Build program failure. (clBuildProgram)\n";
   		return;
   	}
    if(status == CL_OUT_OF_HOST_MEMORY)
   	{
        // if there is a failure to allocate resources required by the
        // OpenCL implementation on the host.
   		std::cout<<"Error: Out of host memory. (clBuildProgram)\n";
   		return;
   	}
    if(status != CL_SUCCESS) 
	{ 
		std::cout<<"Error: Building Program (clBuildProgram)\n";
		return; 
	}

    /* get a kernel object handle for a kernel with the given name */
    kernel = clCreateKernel(program, "edgeDetectKernel", &status);
    if(status != CL_SUCCESS) 
	{  
		std::cout<<"Error: Creating Kernel from program. (clCreateKernel)\n";
		return;
	}
}


// Run OpenCL program
// -> Bind host variables to kernel arguments
// -> Run the CL kernel
double clRunKernels(cl_uint alloc_type, cl_uint kernelCount)
{
    double runTime;

    cl_int   status;
    cl_event events[2];

    size_t globalThreads[1];
    size_t localThreads[1];
    
    globalThreads[0] = kernelCount;
    localThreads[0]  = 1;

    //////////////////////////////////////////
    // Set appropriate arguments to the kernel
    //////////////////////////////////////////

    /* the input array to the kernel */
    status = clSetKernelArg(
                    kernel, 
                    0, 
                    sizeof(cl_mem), 
                    (void *)&inputBuffer);
    if(status != CL_SUCCESS) 
	{ 
		std::cout<<"Error: Setting kernel argument. (input)\n";
		return -1;
	}

    /* the intermediate array to the kernel */
    status = clSetKernelArg(
                    kernel,
                    1,
                    sizeof(cl_mem),
                    (void *)&intermediateBuffer);
    if(status != CL_SUCCESS)
	{
		std::cout<<"Error: Setting kernel argument. (intermediate)\n";
		return -1;
	}

    /* the output array to the kernel */
    status = clSetKernelArg(
                    kernel,
                    2,
                    sizeof(cl_mem), 
                    (void *)&outputBuffer);
    if(status != CL_SUCCESS) 
	{ 
		std::cout<<"Error: Setting kernel argument. (output)\n";
		return -1;
	}

    status = clSetKernelArg(
                    kernel,
                    3,
                    sizeof(cl_mem),
                    (void *)&sobelOpXBuffer);
    if(status != CL_SUCCESS)
	{
		std::cout<<"Error: Setting kernel argument. (sobelx)\n";
		return -1;
	}

    status = clSetKernelArg(
                    kernel,
                    4,
                    sizeof(cl_mem),
                    (void *)&sobelOpYBuffer);
    if(status != CL_SUCCESS)
	{
		std::cout<<"Error: Setting kernel argument. (sobely)\n";
		return -1;
	}

	cl_uint2 inputOutputDim = {width, height};
    status = clSetKernelArg(
                    kernel,
                    5,
                    sizeof(cl_uint2),
                    (void *)&inputOutputDim );
    if(status != CL_SUCCESS)
	{
		std::cout<<"Error: Setting kernel argument. (inputOutputDimensions)\n";
		return -1;
	}

    status = clSetKernelArg(
                    kernel,
                    6,
                    sizeof(cl_uint),
                    (void *)&alloc_type );
    if(status != CL_SUCCESS)
	{
		std::cout<<"Error: Setting kernel argument. (alloc_type)\n";
		return -1;
	}

    sampleCommon.resetTimer(runTimerKey);
    sampleCommon.startTimer(runTimerKey);

    //////////////////////////////////////////
    // Enqueue a kernel run call.
    //////////////////////////////////////////
    status = clEnqueueNDRangeKernel(
			     commandQueue,
                 kernel,
                 1,
                 NULL,
                 globalThreads,
                 localThreads,
                 0,
                 NULL,
                 &events[0]);
    if(status != CL_SUCCESS) 
	{ 
		std::cout<<"Error: Enqueueing kernel onto command queue. (clEnqueueNDRangeKernel)\n";
		return -1;
	}

    //////////////////////////////////////////
    // wait for the kernel call to finish execution
    //////////////////////////////////////////
    status = clWaitForEvents(1, &events[0]);
    if(status != CL_SUCCESS) 
	{ 
		std::cout<<"Error: Waiting for kernel run to finish. (clWaitForEvents 0)\n";
		return -1;
	}

    sampleCommon.stopTimer(runTimerKey);
    runTime = (double)(sampleCommon.readTimer(runTimerKey));

    if (PROFILE)
    {
        long long kernelsStartTime;
        long long kernelsEndTime;

        status = clGetEventProfilingInfo(
                events[0],
                CL_PROFILING_COMMAND_START,
                sizeof(long long),
                &kernelsStartTime,
                NULL);
        if(status != CL_SUCCESS)
    	{
    		std::cout<<"Error: clGetEventProfilingInfo failed (start)\n";
    		return -1;
    	}
        status = clGetEventProfilingInfo(
                events[0],
                CL_PROFILING_COMMAND_END,
                sizeof(long long),
                &kernelsEndTime,
                NULL);
        if(status != CL_SUCCESS)
    	{
    		std::cout<<"Error: clGetEventProfilingInfo failed (end)\n";
    		return -1;
    	}

        /* Compute total time (also convert from nanoseconds to seconds) */
        double totalTime = (double)(kernelsEndTime - kernelsStartTime)/1e9;
        printf("\nTIME: %f\n", totalTime);
        //std::cout<<"TIME: " << totalTime << "\n";
    }

    clReleaseEvent(events[0]);

    //////////////////////////////////////////
    // Enqueue readBuffer
    //////////////////////////////////////////
    status = clEnqueueReadBuffer(
                commandQueue,
                outputBuffer,
                CL_TRUE,
                0,
                width * height * sizeof(cl_uint),
                output,
                0,
                NULL,
                &events[1]);
    if(status != CL_SUCCESS) 
	{ 
        std::cout <<"Error: clEnqueueReadBuffer failed. (clEnqueueReadBuffer)\n";
    }
    
    //////////////////////////////////////////
    // Wait for the read buffer to finish execution
    //////////////////////////////////////////
    status = clWaitForEvents(1, &events[1]);
    if(status != CL_SUCCESS) 
	{ 
		std::cout<<"Error: Waiting for read buffer call to finish. (clWaitForEvents)\n";
		return -1;
	}
    
    clReleaseEvent(events[1]);

    return runTime;
}

// Release OpenCL resources (Context, Memory etc.)
void clCleanup(void)
{
    cl_int status;

    status = clReleaseKernel(kernel);
    if(status != CL_SUCCESS)
	{
		std::cout<<"Error: In clReleaseKernel \n";
		return; 
	}
    status = clReleaseProgram(program);
    if(status != CL_SUCCESS)
	{
		std::cout<<"Error: In clReleaseProgram\n";
		return; 
	}
    status = clReleaseMemObject(inputBuffer);
    if(status != CL_SUCCESS)
	{
		std::cout<<"Error: In clReleaseMemObject (inputBuffer)\n";
		return; 
	}
    status = clReleaseMemObject(intermediateBuffer);
    if(status != CL_SUCCESS)
	{
		std::cout<<"Error: In clReleaseMemObject (intermediateBuffer)\n";
		return;
	}
	status = clReleaseMemObject(outputBuffer);
    if(status != CL_SUCCESS)
	{
		std::cout<<"Error: In clReleaseMemObject (outputBuffer)\n";
		return; 
	}
	status = clReleaseMemObject(sobelOpXBuffer);
    if(status != CL_SUCCESS)
	{
		std::cout<<"Error: In clReleaseMemObject (sobelOpXBuffer)\n";
		return;
	}
	status = clReleaseMemObject(sobelOpYBuffer);
    if(status != CL_SUCCESS)
	{
		std::cout<<"Error: In clReleaseMemObject (sobelOpYBuffer)\n";
		return;
	}
    status = clReleaseCommandQueue(commandQueue);
    if(status != CL_SUCCESS)
	{
		std::cout<<"Error: In clReleaseCommandQueue\n";
		return;
	}
    status = clReleaseContext(context);
    if(status != CL_SUCCESS)
	{
		std::cout<<"Error: In clReleaseContext\n";
		return;
	}
}


// Releases program's resources
void clCleanupHost(void)
{
    if(input != NULL)
    {
        free(input);
        input = NULL;
    }
    if(intermediate != NULL)
    {
        free(input);
        input = NULL;
    }
	if(output != NULL)
	{
		free(output);
		output = NULL;
	}
	if(clSobelOpX != NULL)
	{
		free(output);
		output = NULL;
	}
	if(clSobelOpY != NULL)
	{
		free(output);
		output = NULL;
	}
    if(devices != NULL)
    {
        free(devices);
        devices = NULL;
    }
}

/*Display OpenCL system info */
void clPrintInfo() {
	int MAX_DEVICES = 10;
	size_t p_size;
	size_t arr_tsize[3];
	size_t ret_size;
	char param[100];
	cl_uint entries;
	cl_ulong long_entries;
	cl_bool bool_entries;
	cl_device_id devices[MAX_DEVICES];
	size_t num_devices;
	cl_device_local_mem_type mem_type;
	cl_device_type dev_type;
	cl_device_fp_config fp_conf;
	cl_device_exec_capabilities exec_cap;

	clGetDeviceIDs(	NULL,
					CL_DEVICE_TYPE_DEFAULT,
					MAX_DEVICES,
					devices,
					&num_devices);
	printf("Found Devices:\t\t%d\n", num_devices);

	for (int i = 0; i < num_devices; i++) {
		printf("\nDevice: %d\n\n", i);

		clGetDeviceInfo(devices[i], CL_DEVICE_TYPE, sizeof(dev_type),
				&dev_type, &ret_size);
		printf("\tDevice Type:\t\t");
		if (dev_type & CL_DEVICE_TYPE_GPU)
			printf("CL_DEVICE_TYPE_GPU ");
		if (dev_type & CL_DEVICE_TYPE_CPU)
			printf("CL_DEVICE_TYPE_CPU ");
		if (dev_type & CL_DEVICE_TYPE_ACCELERATOR)
			printf("CL_DEVICE_TYPE_ACCELERATOR ");
		if (dev_type & CL_DEVICE_TYPE_DEFAULT)
			printf("CL_DEVICE_TYPE_DEFAULT ");
		printf("\n");

		clGetDeviceInfo(devices[i], CL_DEVICE_NAME, sizeof(param), param,
				&ret_size);
		printf("\tName: \t\t\t%s\n", param);

		clGetDeviceInfo(devices[i], CL_DEVICE_VENDOR, sizeof(param), param,
				&ret_size);
		printf("\tVendor: \t\t%s\n", param);

		clGetDeviceInfo(devices[i], CL_DEVICE_VENDOR_ID, sizeof(cl_uint),
				&entries, &ret_size);
		printf("\tVendor ID:\t\t%d\n", entries);

		clGetDeviceInfo(devices[i], CL_DEVICE_VERSION, sizeof(param), param,
				&ret_size);
		printf("\tVersion:\t\t%s\n", param);

		clGetDeviceInfo(devices[i], CL_DEVICE_PROFILE, sizeof(param), param,
				&ret_size);
		printf("\tProfile:\t\t%s\n", param);

		clGetDeviceInfo(devices[i], CL_DRIVER_VERSION, sizeof(param), param,
				&ret_size);
		printf("\tDriver: \t\t%s\n", param);

		clGetDeviceInfo(devices[i], CL_DEVICE_EXTENSIONS, sizeof(param), param,
				&ret_size);
		printf("\tExtensions:\t\t%s\n", param);

		clGetDeviceInfo(devices[i], CL_DEVICE_MAX_WORK_ITEM_SIZES, 3
				* sizeof(size_t), arr_tsize, &ret_size);
		printf("\tMax Work-Item Sizes:\t(%d,%d,%d)\n", arr_tsize[0],
				arr_tsize[1], arr_tsize[2]);

		clGetDeviceInfo(devices[i], CL_DEVICE_MAX_WORK_GROUP_SIZE,
				sizeof(size_t), &p_size, &ret_size);
		printf("\tMax Work Group Size:\t%d\n", p_size);

		clGetDeviceInfo(devices[i], CL_DEVICE_MAX_COMPUTE_UNITS,
				sizeof(cl_uint), &entries, &ret_size);
		printf("\tMax Compute Units:\t%d\n", entries);

		clGetDeviceInfo(devices[i], CL_DEVICE_MAX_CLOCK_FREQUENCY,
				sizeof(cl_uint), &entries, &ret_size);
		printf("\tMax Frequency (Mhz):\t%d\n", entries);

		clGetDeviceInfo(devices[i], CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE,
				sizeof(cl_uint), &entries, &ret_size);
		printf("\tCache Line (bytes):\t%d\n", entries);

		clGetDeviceInfo(devices[i], CL_DEVICE_GLOBAL_MEM_SIZE,
				sizeof(cl_ulong), &long_entries, &ret_size);
		printf("\tGlobal Memory (MB):\t%llu\n", long_entries / 1024 / 1024);

		clGetDeviceInfo(devices[i], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong),
				&long_entries, &ret_size);
		printf("\tLocal Memory (MB):\t%llu\n", long_entries / 1024 / 1024);

		clGetDeviceInfo(devices[i], CL_DEVICE_LOCAL_MEM_TYPE,
				sizeof(cl_device_local_mem_type), &mem_type, &ret_size);
		if (mem_type & CL_LOCAL)
			printf("\tLocal Memory Type:\tCL_LOCAL\n");
		else if (mem_type & CL_GLOBAL)
			printf("\tLocal Memory Type:\tCL_GLOBAL\n");
		else
			printf("\tLocal Memory Type:\tUNKNOWN\n");

		clGetDeviceInfo(devices[i], CL_DEVICE_MAX_MEM_ALLOC_SIZE,
				sizeof(cl_ulong), &long_entries, &ret_size);
		printf("\tMax Mem Alloc (MB):\t%llu\n", long_entries / 1024 / 1024);

		clGetDeviceInfo(devices[i], CL_DEVICE_MAX_PARAMETER_SIZE,
				sizeof(size_t), &p_size, &ret_size);
		printf("\tMax Param Size (MB):\t%d\n", p_size);

		clGetDeviceInfo(devices[i], CL_DEVICE_MEM_BASE_ADDR_ALIGN,
				sizeof(cl_uint), &entries, &ret_size);
		printf("\tBase Mem Align (bits):\t%d\n", entries);

		clGetDeviceInfo(devices[i], CL_DEVICE_ADDRESS_BITS, sizeof(cl_uint),
				&entries, &ret_size);
		printf("\tAddress Space (bits):\t%d\n", entries);

		clGetDeviceInfo(devices[i], CL_DEVICE_IMAGE_SUPPORT, sizeof(cl_bool),
				&bool_entries, &ret_size);
		printf("\tImage Support:\t\t%d\n", bool_entries);

		clGetDeviceInfo(devices[i], CL_DEVICE_TYPE, sizeof(fp_conf), &fp_conf,
				&ret_size);
		printf("\tFloat Functionality:\t");
		if (fp_conf & CL_FP_DENORM)
			printf("DENORM support ");
		if (fp_conf & CL_FP_ROUND_TO_NEAREST)
			printf("Round to nearest support ");
		if (fp_conf & CL_FP_ROUND_TO_ZERO)
			printf("Round to zero support ");
		if (fp_conf & CL_FP_ROUND_TO_INF)
			printf("Round to +ve/-ve infinity support ");
		if (fp_conf & CL_FP_FMA)
			printf("IEEE754 fused-multiply-add support ");
		if (fp_conf & CL_FP_INF_NAN)
			printf("INF and NaN support ");
		printf("\n");

		clGetDeviceInfo(devices[i], CL_DEVICE_ERROR_CORRECTION_SUPPORT,
				sizeof(cl_bool), &bool_entries, &ret_size);
		printf("\tECC Support:\t\t%d\n", bool_entries);

		clGetDeviceInfo(devices[i], CL_DEVICE_EXECUTION_CAPABILITIES,
				sizeof(cl_device_exec_capabilities), &exec_cap, &ret_size);
		printf("\tExec Functionality:\t");
		if (exec_cap & CL_EXEC_KERNEL)
			printf("CL_EXEC_KERNEL ");
		if (exec_cap & CL_EXEC_NATIVE_KERNEL)
			printf("CL_EXEC_NATIVE_KERNEL ");
		printf("\n");

		clGetDeviceInfo(devices[i], CL_DEVICE_ENDIAN_LITTLE, sizeof(cl_bool),
				&bool_entries, &ret_size);
		printf("\tLittle Endian Device:\t%d\n", bool_entries);

		clGetDeviceInfo(devices[i], CL_DEVICE_PROFILING_TIMER_RESOLUTION,
				sizeof(size_t), &p_size, &ret_size);
		printf("\tProfiling Res (ns):\t%d\n", p_size);

		clGetDeviceInfo(devices[i], CL_DEVICE_AVAILABLE, sizeof(cl_bool),
				&bool_entries, &ret_size);
		printf("\tDevice Available:\t%d\n", bool_entries);

	}
}

void testEdgeOutput(cl_uint  *out)
{
	for (int y=1; y<height-1; y++)
		for (int x=1; x<width-1; x++)
		{
			int index = (y * width) + x;
			if (out[index] != 1) printf("index[%d]=%d\n",index,out[index]);
		}
}

char* allocTypeToStr(int alloc_type)
{
	char* result = "UNKNOWN";

    switch (alloc_type)
    {
    case ALLOC_TILE:
		 result = "ALLOC_TILE";
		 break;
    case ALLOC_HORZ:
		 result = "ALLOC_HORZ";
		 break;
    case ALLOC_VERT:
		 result = "ALLOC_VERT";
		 break;
    }

    return result;
}

int main(int argc, char * argv[])
{
	clPrintInfo();


	//////////////////////////////
    // Init
    //////////////////////////////
	runTimerKey = sampleCommon.createTimer();
	IplImage* cvImgRaw = cvLoadImage("raw.bmp", 1);
    int repetitions = 100;
    int maxKernels  = 1024;

	width = cvImgRaw->width;
	height = cvImgRaw->height;

	// Initialize both CV and CL code
	clInitializeHost(cvImgRaw);


	//////////////////////////////
    // Serial (OpenCV)
    //////////////////////////////
    sampleCommon.resetTimer(runTimerKey);
    sampleCommon.startTimer(runTimerKey);

    cvGenerateIntensityImage(input,intermediate, height, width);
	cvGenerateSobelImage(intermediate,output, height, width);

    sampleCommon.stopTimer(runTimerKey);

    double cvRunTime = 0;
	for (int run = 0; run < repetitions; run++) {
	    cvRunTime += (double)(sampleCommon.readTimer(runTimerKey));
	}
    printf("\nOpenCV Runtime:\t%f\n\n", cvRunTime/(double)repetitions);

//    IplImage*	cvImgSobel = clArrayToCvImage(output,width,height);
//	cvSaveImage("resultCV.bmp",(CvArr*)cvImgSobel);
//	cvReleaseImage(&cvImgSobel);




	//////////////////////////////
    // Parallel (OpenCL)
    //////////////////////////////
	for (int alloc_type = ALLOC_TILE; alloc_type <= ALLOC_VERT; alloc_type++)
	{
	    printf("\n***********************\n");
		printf("Allocation:\t%s\n", allocTypeToStr(alloc_type));
	    printf("***********************\n");

		for (int kernelCount = 1; kernelCount <= maxKernels; kernelCount = kernelCount * 2)
		{
			printf("Threads[%d]:\t", kernelCount);

			clInitializeHost(cvImgRaw); // Initialize Host application
//			clInitialize(); // Initialize OpenCL resources

			double clRunTime = 0;
			for (int run = 0; run < repetitions; run++) {
				clInitialize(); // Initialize OpenCL resources
				clRunTime += clRunKernels(alloc_type, kernelCount); // Run the CL program
				clCleanup(); // Releases OpenCL resources
			}
			printf("Average Runtime:\t%f\n", clRunTime/(double)repetitions);

//			clCleanup(); // Releases OpenCL resources
			clCleanupHost(); // Release host resources
		}
	}

	cvReleaseImage(&cvImgRaw);

//    IplImage* clSobel = clArrayToCvImage(output,width,height);
//    cvSaveImage("resultCL.bmp",(CvArr*)clSobel);
//	cvReleaseImage(&clSobel);

    return 0;
}
