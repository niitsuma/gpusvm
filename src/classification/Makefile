################################################################################
#
# Build script for project
#
################################################################################

# Add source files here
EXECUTABLE	:= svmClassify
# Cuda source files (compiled with cudacc)
ifeq ($(shared),1)
	CUFILES		:= svmClassify.cu svmClassifyKernels.cu
else
	CUFILES   := svmClassify.cu svmClassifyKernels.cu classifyMain.cu
endif

# C/C++ source files (compiled with gcc / c++)
CCFILES		  :=
LINKCCFILES := svmIO.cpp 
LINKCUFILES := deviceSelect.cu

ifeq ($(emu),1)
        LIB             := -lcublasemu
else
        LIB             := -lcublas -lblas
endif

################################################################################
# Rules and targets

include ../common/common.mk
