#ifndef SVMCLASSIFYKERNELS
#define SVMCLASSIFYKERNELS

#include "../common/framework.h"

#define RBF 0
#define POLYNOMIAL 1
#define LINEAR 2
#define SIGMOID 3
#define UNKNOWN 4

__global__ void makeSelfDots(float* devSource, int devSourcePitchInFloats, float* devDest, int sourceCount, int sourceLength);

__global__ void makeDots(float* devDots, int devDotsPitchInFloats, float* devSVDots, float* devDataDots, int nSV, int nPoints);

__global__ void computeNorms(float* devSV, int devSVPitchInFloats, int nSV, float* devData, int devDataPitchInFloats, int nPoints, float* devNorms, int devNormsPitchInFloats, int nDimension, int nDimensionInBlocks);

__device__ void computeKernels(float* devNorms, int devNormsPitchInFloats, float* devAlphas, int nPoints, int kernelType, int nSV, float coef0, int degree,  float* localAlphas, float* localNorms, int* localFlags);

__device__ void reduce(float mapAValue, float mapBValue, float* localValue, int outputIndex);

__global__ void computeKernelsReduce(float* devNorms, int devNormsPitchInFloats, float* devAlphas, int nPoints, int nSV, int kernelType, float coef0, int degree, float* devLocalValue, int reduceOffset);


__global__ void doClassification(float* devResult, float b, float* devLocalValue, int reduceOffset, int nPoints);

#endif
