#include "svmClassify.h"
#include "svmClassifyKernels.h"

/************
 * This file contains the CUDA functions necessary for SVM Classification
 */
   
/**
 * This function computes self dot products (Euclidean norm squared) for every vector in an array
 * @param devSource the vectors, in column major format
 * @param devSourcePitchInFloats the pitch of each row of the vectors (this is guaranteed to be >= sourceCount.  It might be greater due to padding, to keep each row of the source vectors aligned.
 * @param devDest a vector which will receive the self dot product
 * @param sourceCount the number of vectors
 * @param sourceLength the dimensionality of each vector
 */
__global__ void makeSelfDots(float* devSource, int devSourcePitchInFloats, float* devDest, int sourceCount, int sourceLength) {
	float dot = 0;
	int index = BLOCKSIZE * blockIdx.x + threadIdx.x;

	if (index < sourceCount) {
		for (int i = 0; i < sourceLength; i++) {
			float currentElement = *(devSource + IMUL(devSourcePitchInFloats, i) + index); 
			dot = dot + currentElement * currentElement;
		}
		*(devDest + index) = dot;
	}
}

/**
 * This function constructs a matrix devDots, where devDots_(i,j) = ||data_i||^2 + ||SV_j||^2
 * @param devDots the output array
 * @param devDotsPitchInFloats the pitch of each row of devDots.  Guaranteed to be >= nSV
 * @param devSVDots a vector containing ||SV_j||^2 for all j in [0, nSV - 1]
 * @param devDataDots a vector containing ||data_i||^2 for all i in [0, nPoints - 1]
 * @param nSV the number of Support Vectors in the classifier
 */
__global__ void makeDots(float* devDots, int devDotsPitchInFloats, float* devSVDots, float* devDataDots, int nSV, int nPoints) {
	__shared__ float localSVDots[BLOCKSIZE];
	__shared__ float localDataDots[BLOCKSIZE];
	int svIndex = IMUL(BLOCKSIZE, blockIdx.x) + threadIdx.x;
	
	if (svIndex < nSV) {
		localSVDots[threadIdx.x] = *(devSVDots + svIndex);
	}
	
	int dataIndex = BLOCKSIZE * blockIdx.y + threadIdx.x;
	if (dataIndex < nPoints) {
		localDataDots[threadIdx.x] = *(devDataDots + dataIndex);
	}
	
	__syncthreads();

	dataIndex = BLOCKSIZE * blockIdx.y;
	for(int i = 0; i < BLOCKSIZE; i++, dataIndex++) {
		if ((svIndex < nSV) && (dataIndex < nPoints)) {
			*(devDots + IMUL(devDotsPitchInFloats, dataIndex) + svIndex) = localSVDots[threadIdx.x] + localDataDots[i];
		}
	}
}


__device__ void computeKernels(float* devNorms, int devNormsPitchInFloats, float* devAlphas, int nPoints, int nSV, int kernelType, float coef0, int degree,  float* localValue, int svIndex) {

	if (svIndex < nSV) {
		float alpha = devAlphas[svIndex];
    float norm = devNorms[IMUL(devNormsPitchInFloats, blockIdx.y) + svIndex];
		if(kernelType == RBF)
		{
			localValue[threadIdx.x] = alpha * exp(norm);
		}
		else if(kernelType == LINEAR)
		{
			localValue[threadIdx.x] = alpha * norm;
		}
		else if(kernelType == POLYNOMIAL)
		{
			localValue[threadIdx.x] = alpha * pow(norm + coef0, degree);
		}
		else if(kernelType == SIGMOID)
		{
			localValue[threadIdx.x] = alpha * tanh(norm + coef0);
		}
	}

}



/**
 * This function completes the kernel evaluations and begins the reductions to form the classification result.
 * @param devNorms this contains partially completed kernel evaluations.  For most kernels, devNorms_(i, j) = data_i (dot) sv_j.  For the RBF kernel, devNorms_(i, j) = -gamma*(||data_i||^2 + ||sv_j||^2 - 2* data_i (dot) sv_j)
 * @param devNormsPitchInFloats contains the pitch of the partially completed kernel evaluations.  It will always be >= nSV.
 * @param devAlphas this is the alpha vector for the SVM classifier
 * @param nPoints the number of data points
 * @param nSV the number of support vectors
 * @param kernelType the type of kernel
 * @param coef0 a coefficient used in the polynomial & sigmoid kernels
 * @param degree the degree used in the polynomial kernel
 * @param devLocalValue the local classification results
 * @param reduceOffset computed to begin the reduction properly
 */
__global__ void computeKernelsReduce(float* devNorms, int devNormsPitchInFloats, float* devAlphas, int nPoints, int nSV, int kernelType, float coef0, int degree, float* devLocalValue, int reduceOffset) {
	
	/*Dynamic shared memory setup*/
	
	extern __shared__ float localValue[];
  int svIndex = blockDim.x * blockIdx.x + threadIdx.x;
  
	computeKernels(devNorms, devNormsPitchInFloats, devAlphas, nPoints, nSV, kernelType, coef0, degree, localValue, svIndex);
	__syncthreads();
	
  /*reduction*/
	for(int offset = reduceOffset; offset >= 1; offset = offset >> 1) {
		if ((threadIdx.x < offset) && (svIndex + offset < nSV)) {
			int compOffset = threadIdx.x + offset;
      localValue[threadIdx.x] = localValue[threadIdx.x] + localValue[compOffset];
		}
		__syncthreads();
	}
	if (threadIdx.x == 0) {
		devLocalValue[blockIdx.x + gridDim.x*blockIdx.y] = localValue[0];
	}
}


/*Second stage reduce and cleanup function*/ 
__global__ void doClassification(float* devResult, float b, float* devLocalValue, int reduceOffset, int nPoints) {
	
	extern __shared__ float localValue[];
	
	
  localValue[threadIdx.x] = devLocalValue[blockDim.x*blockIdx.y + threadIdx.x];
  __syncthreads();
  for(int offset = reduceOffset; offset >= 1; offset = offset >> 1) {
    if (threadIdx.x < offset) {
      int compOffset = threadIdx.x + offset;
      if (compOffset < blockDim.x) {
        localValue[threadIdx.x] = localValue[threadIdx.x] + localValue[compOffset];
      }
    }
    __syncthreads();
  }

	float sumResult = localValue[0];
	if (threadIdx.x == 0) {
		sumResult += b;
		devResult[blockIdx.y] = sumResult;
	}
}
