struct MyWork
{
  uint	kernelCount;
  uint	tid;

  uint	width;
  uint	height;

  uint 	localWidth;
  uint 	localHeight;

  uint	x_start;
  uint	x_end;
  uint	y_start;
  uint	y_end;

  uint	edge_x_start;
  uint	edge_x_end;
  uint	edge_y_start;
  uint	edge_y_end;
};

void initTiledWork(struct MyWork* myWork)
{
	uint baseX 		= 0;
	uint baseY 		= 0;

	uint realWidth = myWork->width;
	uint realHeight = myWork->height;
    bool 	powerOf4 	= ( (uint)log2(myWork->kernelCount) % 2) == 0;

    if (!powerOf4)
    {
//    	if (myWork->width > myWork->height)
//    	{
//    		myWork->width = ceil( (float)myWork->width / 2 );
//
//	    	if ( myWork->tid >= (myWork->kernelCount/2) )
//    		{
//	    		baseX	= myWork->width;
//	    		myWork->tid 		= myWork->tid - (myWork->kernelCount/2);
//	    	}
//    	}
//    	else
//    	{
    		myWork->height		= ceil( (float)myWork->height / 2 );

	    	if ( myWork->tid >= (myWork->kernelCount/2) )
    		{
	    		baseY	= myWork->height;
	    		myWork->tid 		= myWork->tid - (myWork->kernelCount/2);
	    	}
//    	}

    	myWork->kernelCount = myWork->kernelCount / 2;
    }

    myWork->localWidth	= ceil( (float)myWork->width  / (float)(sqrt(myWork->kernelCount)) );
    myWork->localHeight	= ceil( (float)myWork->height / (float)(sqrt(myWork->kernelCount)) );

    myWork->x_start		= baseX + ( myWork->tid % (uint)(sqrt(myWork->kernelCount)) ) * myWork->localWidth;
    myWork->x_end		= baseX + min( myWork->x_start + myWork->localWidth , myWork->width);
    myWork->y_start		= baseY + ( myWork->tid / (uint)(sqrt(myWork->kernelCount)) ) * myWork->localHeight;
    myWork->y_end		= baseY + min( myWork->y_start + myWork->localHeight , myWork->height);

    myWork->edge_x_start	= (myWork->x_start > 0) ? myWork->x_start : myWork->x_start + 1;
    myWork->edge_x_end		= (myWork->x_end < realWidth) ? myWork->x_end : myWork->x_end - 1;
    myWork->edge_y_start 	= (myWork->y_start > 0)	? myWork->y_start : myWork->y_start + 1;
    myWork->edge_y_end		= (myWork->y_end < realHeight) ? myWork->y_end : myWork->y_end - 1;
}

void initHorizWork(struct MyWork* myWork)
{
    myWork->localWidth	= myWork->width;
    myWork->localHeight	= ceil( (float)myWork->height / (float)(myWork->kernelCount) );

    myWork->x_start		= 0;
    myWork->x_end		= myWork->width;
    myWork->y_start		= myWork->tid * myWork->localHeight;
    myWork->y_end		= min( myWork->y_start + myWork->localHeight , myWork->height);

    myWork->edge_x_start	= 1;
    myWork->edge_x_end		= myWork->x_end - 1;
    myWork->edge_y_start 	= (myWork->y_start > 0)	? myWork->y_start : myWork->y_start + 1;
    myWork->edge_y_end		= (myWork->y_end < myWork->height) ? myWork->y_end : myWork->y_end - 1;
}

void initVertWork(struct MyWork* myWork)
{
    myWork->localWidth	= ceil( (float)myWork->width / (float)(myWork->kernelCount) );
    myWork->localHeight	= myWork->height;

    myWork->x_start		= myWork->tid * myWork->localWidth;
    myWork->x_end		= min( myWork->x_start + myWork->localWidth , myWork->width);
    myWork->y_start		= 0;
    myWork->y_end		= myWork->height;

    myWork->edge_x_start	= (myWork->x_start > 0)	? myWork->x_start : myWork->x_start + 1;
    myWork->edge_x_end		= (myWork->x_end < myWork->width) ? myWork->x_end : myWork->x_end - 1;
    myWork->edge_y_start 	= 1;
    myWork->edge_y_end		= myWork->y_end - 1;
}

__kernel void edgeDetectKernel(	__global  	uint4 * input,
								__global  	uint  * intermediate,
								__global  	uint  * output,
                                __global  	uint  * clSobelOpX,
                                __global  	uint  * clSobelOpY,
                                const     	uint2 	inputOutputDim,
                                const     	uint 	alloc_type
                                )
{
	struct MyWork myWork;
	myWork.tid			= get_global_id(0);
	myWork.kernelCount	= get_global_size(0);
	myWork.width 		= inputOutputDim.x;
	myWork.height		= inputOutputDim.y;

	 switch (alloc_type)
	 {
	 case 0:	//TILE
		 initTiledWork( &myWork );
		 break;
	 case 1:	//HORIZONTAL
		 initHorizWork( &myWork );
		 break;
	 case 2:	//VERTICAL
		 initVertWork( &myWork );
		 break;
	 }

	int	y;
	int	x;
	int	index;

	for (y = myWork.y_start; y < myWork.y_end; y++)
		for (x = myWork.x_start; x < myWork.x_end; x++)
		{
			index = (y * inputOutputDim.x) + x;
			intermediate[index] = (input[index].x + input[index].y + input[index].z)/3;
//			output[index] = intermediate[index];
//			output[index] = ((float)get_global_id(0)/(float)kernelCount)*(float)(255.0);
		}

//	barrier(CLK_GLOBAL_MEM_FENCE);
	mem_fence(CLK_GLOBAL_MEM_FENCE);

	int Gx;
	int Gy;

	for (y = myWork.edge_y_start; y < myWork.edge_y_end; y++)
	{
		for (x = myWork.edge_x_start; x < myWork.edge_x_end; x++)
		{
		index = (y * inputOutputDim.x) + x;

		Gx = 	(intermediate[index - inputOutputDim.x - 1] 	*   clSobelOpX[0]) +
				(intermediate[index - inputOutputDim.x] 		*   clSobelOpX[1]) +
				(intermediate[index - inputOutputDim.x + 1]		*   clSobelOpX[2]) +

				(intermediate[index - 1]						*   clSobelOpX[3]) +
				(intermediate[index]							*   clSobelOpX[4]) +
				(intermediate[index + 1]						*   clSobelOpX[5]) +

				(intermediate[index + inputOutputDim.x - 1]		*   clSobelOpX[6]) +
				(intermediate[index + inputOutputDim.x]			*   clSobelOpX[7]) +
				(intermediate[index + inputOutputDim.x + 1]		*   clSobelOpX[8]);
		Gx =	abs(Gx);

		Gy = 	(intermediate[index - inputOutputDim.x - 1] 	*   clSobelOpY[0]) +
				(intermediate[index - inputOutputDim.x] 		*   clSobelOpY[1]) +
				(intermediate[index - inputOutputDim.x + 1]		*   clSobelOpY[2]) +

				(intermediate[index - 1]						*   clSobelOpY[3]) +
				(intermediate[index]							*   clSobelOpY[4]) +
				(intermediate[index + 1]						*   clSobelOpY[5]) +

				(intermediate[index + inputOutputDim.x - 1]		*   clSobelOpY[6]) +
				(intermediate[index + inputOutputDim.x]			*   clSobelOpY[7]) +
				(intermediate[index + inputOutputDim.x + 1]		*   clSobelOpY[8]);
		Gy =	abs(Gy);

		output[index] = Gx+Gy;
	}
}

}
