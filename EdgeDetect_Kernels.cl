__kernel void edgeDetectKernel(	__global  	uint4 * input,
								__global  	uint  * intermediate,
								__global  	uint  * output,
                                __global  	uint  * clSobelOpX,
                                __global  	uint  * clSobelOpY,
								__private	uint	kernelCount,
                                const     	uint2 	inputOutputDim
                                )
{	
    uint 	tid 		= get_global_id(0);
    uint	baseX 		= 0;
    uint	baseY 		= 0;
    bool 	powerOf4 	= ( (uint)log2(kernelCount) % 2) == 0;
    uint	width 		= inputOutputDim.x;
    uint	height		= inputOutputDim.y;

//    float 	tempTid 		= tid;
//    float 	tempKernelCount = kernelCount;

    if (!powerOf4)
    {
    	if (width > height)
    	{
    		width		= ceil( (float)width / 2 );

	    	if ( tid >= (kernelCount/2) )
    		{
	    		baseX	= width;
	    		tid 	= tid - (kernelCount/2);
	    	}
    	}
    	else
    	{
    		height		= ceil( (float)height / 2 );

	    	if ( tid >= (kernelCount/2) )
    		{
	    		baseY	= height;
	    		tid 	= tid - (kernelCount/2);
	    	}
    	}

    	kernelCount = kernelCount / 2;
    }

    uint 	localWidth	= ceil( (float)width  / (float)(sqrt(kernelCount)) );
    uint 	localHeight	= ceil( (float)height / (float)(sqrt(kernelCount)) );

	uint	x_start		= baseX + ( tid % (uint)(sqrt(kernelCount)) ) * localWidth;
	uint	x_end		= baseX + min( x_start + localWidth , width);
	uint	y_start		= baseY + ( tid / (uint)(sqrt(kernelCount)) ) * localHeight;
	uint	y_end		= baseY + min( y_start + localHeight , height);

	int	y;
	int	x;
	int	index;

	for (y = y_start; y < y_end; y++)
	{
		for (x = x_start; x < x_end; x++)
		{
			index = (y * inputOutputDim.x) + x;
			intermediate[index] = (input[index].x + input[index].y + input[index].z)/3;
//			output[index] = intermediate[index];
//			output[index] = (tempTid/tempKernelCount)*255;
		}
	}

//	barrier(CLK_GLOBAL_MEM_FENCE);
	mem_fence(CLK_GLOBAL_MEM_FENCE);

	int	edge_x_start 	= (x_start > 0) 					? x_start : x_start + 1;
	int	edge_x_end		= (x_end   < inputOutputDim.x) 		? x_end   : x_end - 1;
	int	edge_y_start 	= (y_start > 0)						? y_start : y_start + 1;
	int	edge_y_end		= (y_end   < inputOutputDim.y) 		? y_end   : y_end - 1;

//	printf("\n[%d] start[%d,%d] edge_start[%d,%d] end[%d,%d] edge_end[%d,%d]\n",tid,
//																				x_start,y_start,edge_x_start,edge_y_start,
//																				x_end,y_end,edge_x_end,edge_y_end);
//	printf("\n[%d] edge_start[%d,%d] edge_end[%d,%d]\n",tid,
//														edge_x_start,edge_y_start,edge_x_end,edge_y_end);

	int Gx;
	int Gy;
	int G;

		for (y = edge_y_start; y < edge_y_end; y++)
		{
			for (x = edge_x_start; x < edge_x_end; x++)
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

			G = Gx+Gy;

			output[index] = G;
		}
	}

}
