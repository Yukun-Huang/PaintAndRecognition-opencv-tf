#include "python_tf_minst.h"

using namespace std;
using namespace cv;

// global variables ///////////////////////////////////////////////////////////////////////////////
const int MIN_CONTOUR_AREA = 100;

const int RESIZED_IMAGE_WIDTH = 20;
const int RESIZED_IMAGE_HEIGHT = 30;

/************************************************
 *            类：ContourWithData
 * 成员变量：轮廓(点集)，最小包围矩形(Rect)，轮廓面积(float)
 * 成员函数：
 * checkIfContourIsValid() 如果轮廓面积小于最小阈值，则认为该轮廓不合理，返回0,否则返回1
 * sortByBoundingRectXPosition() 如果左轮廓包围矩形x坐标小于右轮廓包围矩形x坐标，返回1,否则返回0
/************************************************/
class ContourWithData {
public:
    // member variables ///////////////////////////////////////////////////////////////////////////
    std::vector<cv::Point> ptContour;           // contour
    cv::Rect boundingRect;                      // bounding rect for contour
    float fltArea;                              // area of contour

    ///////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////
    bool checkIfContourIsValid() {                              // obviously in a production grade program
        if (fltArea < MIN_CONTOUR_AREA) return false;           // we would have a much more robust function for
        return true;                                            // identifying if a contour is valid !!
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    static bool sortByBoundingRectXPosition(const ContourWithData& cwdLeft, const ContourWithData& cwdRight) {      // this function allows us to sort
        return(cwdLeft.boundingRect.x < cwdRight.boundingRect.x);                                                   // the contours from left to right
    }

};

Python_TF_MINST::Python_TF_MINST()
{

}

void Python_TF_MINST::Python_Init()
{
    Py_Initialize();
    if( !Py_IsInitialized() )
    {
        std::cout << "Python initialized failed!" << std::endl;
        return;
    }
}

int Python_TF_MINST::Python_LoadModuleAndFunction()
{
    //添加当前路径
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append('../python/')");

    // 载入名为predict的脚本
    string pyname = "predict_cnn";
    pName = PyString_FromString(pyname.data());
    pModule = PyImport_Import(pName);
    if ( !pModule )
    {
        cout << "can't find " + pyname + ".py" << endl;
        getchar();
        return -1;
    }
    pDict = PyModule_GetDict(pModule);
    if ( !pDict )
    {
        printf("PyModule_GetDict failed!");
        getchar();
        return -1;
    }

    // 找出函数名为predictint的函数
    pFunc = PyDict_GetItemString(pDict, "predictint");
    if ( !pFunc || !PyCallable_Check(pFunc) )
    {
        printf("can't find function [predictint]");
        getchar();
        return -1;
    }
}

void Python_TF_MINST::Opencv_imagePrepare(cv::Mat image)
{
    if (image.empty()) {                                // if unable to open image
        std::cout << "error: image not read from file\n\n";         // show error message on command line
        return;                                                  // and exit program
    }

    std::vector<ContourWithData> allContoursWithData;           // declare empty vectors,
    std::vector<ContourWithData> validContoursWithData;         // we will fill these shortly

    cv::Mat matGrayscale;           //
    cv::Mat matBlurred;             // declare more image variables
    cv::Mat matThresh;              //
    cv::Mat matThreshCopy;          //

    cv::cvtColor(image, matGrayscale, CV_BGR2GRAY);         // convert to grayscale

    cv::GaussianBlur(matGrayscale,              // input image
                     matBlurred,                // output image
                     cv::Size(5, 5),            // smoothing window width and height in pixels
                     0);                        // sigma value, determines how much the image will be blurred, zero makes function choose the sigma value

                                            // filter image from grayscale to black and white
//    cv::adaptiveThreshold(matBlurred,                           // input image
//                          matThresh,                            // output image
//                          255,                                  // make pixels that pass the threshold full white
//                          cv::ADAPTIVE_THRESH_GAUSSIAN_C,       // use gaussian rather than mean, seems to give better results
//                          cv::THRESH_BINARY_INV,                // invert so foreground will be white, background will be black
//                          11,                                   // size of a pixel neighborhood used to calculate threshold value
//                          2);                                   // constant subtracted from the mean or weighted mean
    matThresh = matBlurred > 1;
    matThreshCopy = matThresh.clone();              // make a copy of the thresh image, this in necessary b/c findContours modifies the image

    std::vector<std::vector<cv::Point> > ptContours;        // declare a vector for the contours
    std::vector<cv::Vec4i> v4iHierarchy;                    // declare a vector for the hierarchy (we won't use this in this program but this may be helpful for reference)

    cv::findContours(matThreshCopy,             // input image, make sure to use a copy since the function will modify this image in the course of finding contours
        ptContours,                             // output contours
        v4iHierarchy,                           // output hierarchy
        cv::RETR_EXTERNAL,                      // retrieve the outermost contours only
        cv::CHAIN_APPROX_SIMPLE);               // compress horizontal, vertical, and diagonal segments and leave only their end points

    for (int i = 0; i < ptContours.size(); i++) {               // for each contour
        ContourWithData contourWithData;                                                    // instantiate a contour with data object
        contourWithData.ptContour = ptContours[i];                                          // assign contour to contour with data
        contourWithData.boundingRect = cv::boundingRect(contourWithData.ptContour);         // get the bounding rect
        contourWithData.fltArea = cv::contourArea(contourWithData.ptContour);               // calculate the contour area
        allContoursWithData.push_back(contourWithData);                                     // add contour with data object to list of all contours with data
    }

    for (int i = 0; i < allContoursWithData.size(); i++) {                      // for all contours
        if (allContoursWithData[i].checkIfContourIsValid()) {                   // check if valid
            validContoursWithData.push_back(allContoursWithData[i]);            // if so, append to valid contour list
        }
    }
            // sort contours from left to right
    std::sort(validContoursWithData.begin(), validContoursWithData.end(), ContourWithData::sortByBoundingRectXPosition);

    for (int i = 0; i < validContoursWithData.size(); i++) {            // for each contour
                                                                    // draw a green rect around the current char
        cv::rectangle(image,                            // draw rectangle on original image
                      validContoursWithData[i].boundingRect,        // rect to draw
                      cv::Scalar(0, 255, 0),                        // green
                      2);                                           // thickness

        cv::Mat matROI = matThresh(validContoursWithData[i].boundingRect);          // get ROI image of bounding rect
        cv::Mat matROIResized;
        // tensorflow's model-need prepare:
        float width  = float(matROI.cols);
        float height = float(matROI.rows);
        if(width > height)
        {
            int nheight = round(20.0/width*height);
            if (nheight == 0)
                nheight = 1;
            cv::resize(matROI, matROIResized, cv::Size(20, nheight)); // resize image, this will be more consistent for recognition and storage
            cv::copyMakeBorder(matROIResized,matROIResized,round((28-nheight)/2),round((28-nheight)/2),4,4,cv::BORDER_CONSTANT,Scalar(0));
        }
        else
        {
            int nwidth = round(20.0/height*width);
            if (nwidth == 0)
                nwidth = 1;
            cv::resize(matROI, matROIResized, cv::Size(nwidth, 20)); // resize image, this will be more consistent for recognition and storage
            cv::copyMakeBorder(matROIResized,matROIResized,4,4,round((28-nwidth)/2),round((28-nwidth)/2),cv::BORDER_CONSTANT,Scalar(0));
        }

        Mat kernel = (Mat_<float>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
        cv::filter2D(matROIResized, matROIResized, matROIResized.depth(), kernel);

        cv::Mat matROIFloat;
        matROIResized.convertTo(matROIFloat, CV_32FC1);             // convert Mat to float, necessary for call to find_nearest
        imshow("matROIResized",matROIResized);
        imwrite("sam.jpeg",~matROIResized);
//        cout << "one way:" << endl;
//        cv::Mat matROIFlattenedFloat = matROIFloat.reshape(1, 1);
//        cv::normalize(matROIFlattenedFloat,matROIFlattenedFloat,0,1,NORM_MINMAX);

        cout << "another way:" << endl;
        uchar *pt = matROIResized.ptr<uchar>(0);
        for(int i=0,j=0;i<28*28;i++)
        {
            pt_featureVec[i] = int(*pt++)/255.0f;
            //cout << pt_featureVec[i] << "  ";if(j==28){j=0;cout << endl;}j++;
        }
        Python_Run(pt_featureVec);
    }
}

void Python_TF_MINST::Python_Run(float *featureVec)
{
    // 参数进栈
    *pArgs;
    pArgs = PyTuple_New(1);
    // PyObject* arg1 = Py_BuildValue("i", 100); // 整数参数
    // PyObject* arg2 = Py_BuildValue("f", 3.14); // 浮点数参数
    // PyObject* arg3 = Py_BuildValue("s", "hello"); // 字符串参数
    // PyTuple_SetItem(args, 0, arg1);

    //  PyObject* Py_BuildValue(char *format, ...)
    //  把C++的变量转换成一个Python对象。当需要从
    //  C++传递变量到Python时，就会使用这个函数。此函数
    //  有点类似C的printf，但格式不同。常用的格式有
    //  s 表示字符串，
    //  i 表示整型变量，
    //  f 表示浮点数，
    //  O 表示一个Python对象。

    // PyArg_ParseTuple(pRetValue, "i", &ret);
    //    for(int j = 0; j < PyList_Size(pRetValue); j++)
    //    {
    //        PyObject *pValue = PyList_GetItem(pRetValue, j);
    //    }

    // 设置参数
    PyObject *pFeatureList = PyList_New(28*28);
    for(int i = 0; i<784; i++)
        PyList_SetItem(pFeatureList,i,Py_BuildValue("f",featureVec[i]));
        PyTuple_SetItem(pArgs, 0, pFeatureList);

    // 调用Python函数
    pRetValue = PyObject_CallObject(pFunc, pArgs);
    // 变量转换
    int ret = 10;
    PyErr_Occurred();PyErr_Print();
    PyArg_ParseTuple(pRetValue, "i", &ret);
    if (pRetValue)
    {
        printf("number is %d.\n", ret);
    }


}

void Python_TF_MINST::Python_Final()
{
    Py_DECREF(pName);
    Py_DECREF(pArgs);
    Py_DECREF(pModule);

    // 关闭Python，与Py_Initialize()对应
    Py_Finalize();
}

int Python_TF_MINST::test()
{
    //添加当前路径
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append('../python/')");
    //PyRun_SimpleString("import tensorflow");
    //PyRun_SimpleString("print 'Hello'");
    // 载入名为predict_test的脚本
    pName = PyString_FromString("predict_test");
    pModule = PyImport_Import(pName);
    if ( !pModule )
    {
        printf("can't find predict_test.py");
        getchar();
        return -1;
    }
    pDict = PyModule_GetDict(pModule);
    if ( !pDict )
    {
        printf("PyModule_GetDict failed!");
        getchar();
        return -1;
    }
    //尝试调用Hello函数
    pFunc = PyDict_GetItemString(pDict, "Hello");
    if ( !pFunc || !PyCallable_Check(pFunc) )
    {
        printf("can't find function [Hello]");
        getchar();
        return -1;
    }
    PyObject_CallObject(pFunc, NULL);
    // 找出函数名为main的函数
    pFunc = PyDict_GetItemString(pDict, "main");
    if ( !pFunc || !PyCallable_Check(pFunc) )
    {
        printf("can't find function [main]");
        getchar();
        return -1;
    }

    *pArgs;
    pArgs = PyTuple_New(1);

    //  PyObject* Py_BuildValue(char *format, ...)
    //  把C++的变量转换成一个Python对象。当需要从
    //  C++传递变量到Python时，就会使用这个函数。此函数
    //  有点类似C的printf，但格式不同。常用的格式有
    //  s 表示字符串，
    //  i 表示整型变量，
    //  f 表示浮点数，
    //  O 表示一个Python对象。

    PyTuple_SetItem(pArgs, 0, Py_BuildValue("s","num4.jpeg"));

    // 调用Python函数
    pRetValue = PyObject_CallObject(pFunc, pArgs);
    long r1= PyInt_AsLong(pRetValue);
    //PyArg_ParseTuple(pRetValue, "i", &r1);
    if (pRetValue)
    {
        printf("According to python retValue: number is %ld.\n", r1);
    }
}
