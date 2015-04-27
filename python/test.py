#!/usr/bin/env python

import subprocess
import shlex
import csv
import sys
import time
from sklearn.base import BaseEstimator,RegressorMixin, ClassifierMixin
from sklearn.datasets import load_svmlight_file,dump_svmlight_file

import libsvm
import libsvm.svmutil
import numpy as np
import scipy.sparse
import scipy

from svm2classcuda import *

    
def _debug():
    import os.path
    import numpy as np
    trainX_fname='mytrainX.npy'
    trainY_fname='mytrainY.npy'
    taindat_fname='mytrain.dat'
    if os.path.exists(trainX_fname) :
        X=np.load(trainX_fname)
        Y=np.load(trainY_fname)
        X=X.tolist()
        Y=Y.tolist()
    else:
        import random
        N=20
        #N=400
        dim=4
        X=[[random.random() for m in range(dim)] for n in range(N)]
        ###Y=[random.choice([-1,1]) for n in range(N)]
        Y=[1 if np.linalg.norm(x) > np.linalg.norm(np.ones(dim)*0.5) else -1 for x  in X] ##2class
        np.save(trainX_fname,X)
        np.save(trainY_fname,Y)
        #dump_svmlight_file(X,Y,taindat_fname ,zero_based=False)
        dump_svmlight_file_dense(X,Y,taindat_fname ,zero_based=False)
    print np.array(X)
    print Y


    print 'libsvm python ffi----------------------------'
    import libsvm
    import libsvm.svmutil
    params = libsvm.svmutil.svm_parameter('-t 2 -c 3 -g 0.7')
    print params
    study_data = libsvm.svmutil.svm_problem(Y, X)    
    model = libsvm.svmutil.svm_train(study_data, params)
    # #model = libsvm.svmutil.svm_train(Y,X, params)
    print model
    Ypred_libsvmffi = libsvm.svmutil.svm_predict(Y,X , model)
    print map(int,Ypred_libsvmffi[0])
    
    ### sklearn
    print 'sklean -------------------'
    import numpy as np

    X = np.array(X)
    Y = np.array(Y)
    from sklearn.svm import SVC
    clf=SVC(C=3.0, gamma=0.7, kernel='rbf')
    clf.fit(X, Y)

    Ypred_sklearnSVC=clf.predict(X)
    print  Ypred_sklearnSVC.tolist()
    from sklearn.metrics import accuracy_score
    print accuracy_score(Y, Ypred_sklearnSVC)
    
    print 'cmdline wrap-------------------'
    clfcmd=ClassifySVM2ClassCuda(C=3.0, gamma=0.7, kernel='rbf')    
    clfcmd.fit(X, Y)
    Ypred_cmdwrap=clfcmd.predict(X)
    print Ypred_cmdwrap.tolist()
    print Ypred_sklearnSVC.tolist()
    print Y.tolist()



if __name__ == '__main__':
    _debug()

    
    
