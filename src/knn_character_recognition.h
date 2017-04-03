#ifndef KNN_CHARACTER_RECOGNITION_H
#define KNN_CHARACTER_RECOGNITION_H

#include <iostream>
#include <opencv2/opencv.hpp>

class KNN_Character_Recognition
{
private:
    cv::Mat matClassificationInts;                //label
    cv::Mat matTrainingImagesAsFlattenedFloats;   //feature

public:
    cv::Ptr<cv::ml::KNearest>  kNearest;
    cv::Ptr<cv::ml::TrainData> trainingData;

    KNN_Character_Recognition();
    void readDataSet();
    void train();
    void classify(cv::Mat matTestingNumbers,std::string &strFinalString);
};


#endif // KNN_CHARACTER_RECOGNITION_H
