#ifndef SVMCLASSIFY
#define SVMCLASSIFY

#include "svmCommon.h"

/**
 * Performs SVM classification.
 * @param data the data to be classfied, stored as a flat column major array.
 * @param nData the number of data points being classified
 * @param supportVectors the support vectors of the classifier, stored as a flat column major array.
 * @param nSV the number of support vectors of the classifier
 * @param nDimension the dimensionality of the data and support vectors
 * @param kp a struct containing all the information about the kernel parameters
 * @param p_result a pointer to a float pointer where the results will be placed.  The perform classification routine will allocate the output buffer.
 */
void performClassification(float *data, int nData, float *supportVectors, int nSV, int nDimension, float* alpha, Kernel_params kp, float** p_result);

#endif
