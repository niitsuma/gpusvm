################################################################################
#
# Build script for project
#
################################################################################

# Add source files here
#EXECUTABLE	:= svmClassify
EXECUTABLE	:= svm2classcudapredict
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


LIB_OPENBLAS_OPENMP := $(strip $(wildcard /usr/lib/libopenblas_openmp.so))
LIB_OPENBLAS         := $(strip $(wildcard /usr/lib/libopenblas.so))
ifneq ($(LIB_OPENBLAS_OPENMP),)
    LIBBLAS = "-lopenblas_openmp"
else ifneq  ($(LIB_OPENBLAS),)
    LIBBLAS = "-lopenblas"
else
    LIBBLAS = "-lblas"
endif


ifeq ($(emu),1)
        LIB             := -lcublasemu
else
        #LIB             := -lcublas -lblas
        LIB             := -lcublas $(LIBBLAS)
endif

################################################################################
# Rules and targets

include ../common/common.mk
