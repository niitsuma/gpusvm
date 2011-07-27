################################################################################
#
# Build script for project
#
################################################################################

# Add source files here
EXECUTABLE	:= svmTrain
CUFILES   := trainingMain.cu svmTrain.cu
# C/C++ source files (compiled with gcc / c++)
CCFILES		:= Cache.cpp Controller.cpp
LINKCCFILES := svmIO.cpp
LINKCUFILES := deviceSelect.cu


################################################################################
# Rules and targets

include ../common/common.mk
