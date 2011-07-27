/* Includes, system */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
/* Includes, cuda */
#include "cublas.h"
#include "cutil.h"
#include "cuda.h"

/* Includes, project */
#include "../common/framework.h"
#include "svmClassify.h"
#include "../common/svmIO.h"


void printHelp() {
  printf("Usage: svmClassify modelFile dataFile [outputFile]\n");
}

/**
 * This main function performs SVM classification from a file.
 * It expects that the first command line argument is a model file (in the same format as LibSVM) and the second argument is a data file.
 */
int main( const int argc, const char** argv) {


	int nSV;		//total number of support vectors
	int total_nPoints;	//total number of test points
	int nDimension;		//data dimension
	float* alpha;		//alpha array
	float* supportVectors;	//support vector data
	int dataDimension;	//dimension in data (must be equal to nDimension)
	float* labels;		//labels for the test data (for measuring accuracy etc.)
	float* data;		//test data

	struct Kernel_params kp;

  if (argc < 3) {
    printHelp();
    return(0);
  }
  
	float class1Label, class2Label;
	int success = readModel(argv[1], &alpha, &supportVectors, &nSV, &nDimension, &kp, &class1Label, &class2Label);
	if (success == 0) {
		printf("Invalid Model\n");
		exit(1);
	}
	

	success = readSvm(argv[2], &data, &labels, &total_nPoints, &dataDimension);
	if (success == 0) {
		printf("Invalid Data\n");
		exit(2);
	}
	if (dataDimension != nDimension) {
		printf("This data isn't compatible with this model\n");
		exit(3);
	}
  char* outputFilename;
  if (argc == 4) {
    outputFilename = (char*)malloc(sizeof(char)*(strlen(argv[3])));
    strcpy(outputFilename, argv[3]);
 
  } else {
    int inputNameLength = strlen(argv[2]);
    outputFilename = (char*)malloc(sizeof(char)*(inputNameLength + 5));
    strncpy(outputFilename, argv[2], inputNameLength + 4);
    char* period = strrchr(outputFilename, '.');
    if (period == NULL) {
      period = outputFilename + inputNameLength;
    }
    strncpy(period, ".dat\0", 5); 
  }
	
	printf("Model found: %d support vectors\n", nSV);
	printf("Data found: %d points\n", total_nPoints);
	printf("Problem is %d dimensional\n", nDimension);
  printf("Output file: %s\n", outputFilename);
  struct timeval start;
  gettimeofday(&start,0);
	
	
	float * result;
	performClassification(data, total_nPoints, supportVectors, nSV, nDimension, alpha, kp, &result);	
	struct timeval finish;
  gettimeofday(&finish, 0);
  float classificationTime = (float)(finish.tv_sec - start.tv_sec) + ((float)(finish.tv_usec - start.tv_usec)) * 1e-6;
	
	printf("Classification time : %f seconds\n", classificationTime);
  int confusionMatrix[] = {0, 0, 0, 0};
	for (int i = 0; i < total_nPoints; i++) {
		if ((labels[i] == class2Label) && (result[i] < 0)) {
			confusionMatrix[0]++;
		} else if ((labels[i] == class2Label) && (result[i] >= 0)) {
			confusionMatrix[1]++;
		} else if ((labels[i] == class1Label) && (result[i] < 0)) {
			confusionMatrix[2]++;
		} else if ((labels[i] == class1Label) && (result[i] >= 0)) {
			confusionMatrix[3]++;
		}
	}
	printf("Accuracy: %f (%d / %d) \n", (float)(confusionMatrix[0] + confusionMatrix[3])*100.0/((float)total_nPoints),confusionMatrix[0]+confusionMatrix[3], total_nPoints);
  printClassification(outputFilename, result, total_nPoints);
	
}
	
