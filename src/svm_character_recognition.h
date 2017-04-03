#ifndef SVM_CHARACTER_RECOGNITION_H
#define SVM_CHARACTER_RECOGNITION_H

#include <iostream>
#include <opencv2/opencv.hpp>

class SVM_Character_Recognition
{
private:
    cv::Mat matClassificationInts;                //label
    cv::Mat matTrainingImagesAsFlattenedFloats;   //feature

public:
    cv::Ptr<cv::ml::SVM>  svm;
    cv::Ptr<cv::ml::TrainData> trainingData;

    SVM_Character_Recognition();
    void readDataSet();
    void train();
    void classify(cv::Mat matTestingNumbers,std::string &strFinalString);
};


#endif // SVM_CHARACTER_RECOGNITION_H
