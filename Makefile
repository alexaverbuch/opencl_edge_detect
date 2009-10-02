DEPTH = ../../../../..

include $(DEPTH)/make/openclsdkdefs.mk 

####
#
#  Targets
#
####

CL_SAMPLE_EXE				= 1
INSTALL_TO_PUBLIC       	= 1
EXE_TARGET 					= EdgeDetect
EXE_TARGET_INSTALL   		= EdgeDetect
SRC_TARGET_INSTALL      	= EdgeDetect.cpp EdgeDetect.hpp EdgeDetect_Kernels.cl EdgeDetect.vcproj
SRC_DIR_INSTALL				= samples/opencl/cl/app/EdgeDetect/

####
#
#  CPP files
#
####

CPPFILES 	= EdgeDetect.cpp 
CLFILES		= EdgeDetect_Kernels.cl
LLIBS  		+= SDKUtil cxcore cv highgui cvaux ml
INCLUDEDIRS += $(SDK_HEADERS) 

include $(DEPTH)/make/openclsdkrules.mk 

