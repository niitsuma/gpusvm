#include <vector>
#include <sys/time.h>
#include "svmCommon.h"
#include <cstdlib>
using std::vector;

class Controller {
 public:
  Controller(float initialGap, SelectionHeuristic currentMethodIn, int samplingIntervalIn, int problemSize);
  void addIteration(float gap);
  void print();
  SelectionHeuristic getMethod();
 private:
  bool adaptive;
  int samplingInterval;
  vector<float> progress;
  vector<int> method;
  SelectionHeuristic currentMethod;
  vector<float> rates;
  int timeSinceInspection;
  int inspectionPeriod;
  int beginningOfEpoch;
  int middleOfEpoch;
  int currentInspectionPhase;
  float filter(int begin, int end);
  float findRate(struct timeval* start, struct timeval* finish, int beginning, int end);
  struct timeval start;
  struct timeval mid;
  struct timeval finish;
};
