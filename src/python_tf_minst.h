#ifndef PYTHON_TF_MINST_H
#define PYTHON_TF_MINST_H

#include <Python.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <math.h>

class Python_TF_MINST
{
private:
    PyObject *pName,*pModule,*pDict,*pFunc,*pArgs;
    float *pt_featureVec = new float[28*28];
public:
    PyObject * pRetValue;   //函数返回值
public:
    Python_TF_MINST();
    void Python_Init();
    int  Python_LoadModuleAndFunction();
    void Opencv_imagePrepare(cv::Mat);
    void Python_Run(float*);
    void Python_Final();
    int test();
};



#endif // PYTHON_TF_MINST_H
