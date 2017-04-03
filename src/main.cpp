#include <iostream>
#include <fstream>
#include <stdio.h>
#include <algorithm>
//#include <windows.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "kcftracker.hpp"
#include "knn_character_recognition.h"
#include "svm_character_recognition.h"
#include "python_tf_minst.h"

using namespace std;
using namespace cv;

#define THICKNESS 8

//函数声明
void MouseProcess(Rect& trackBox,bool clickFlag);
bool ROIProcess(Rect& trackBox);
//flag
bool HOG = true;
bool FIXEDWINDOW = false;
bool MULTISCALE = true;
bool SILENT = true;
bool LAB = false;
bool PAUSE = false;

bool selectObject = false;
bool trackObject = false;

bool clickFlag = false;

int color_num = 0;
Scalar color(0,0,255); //默认红色

// Create KCFTracker object
KCFTracker tracker(HOG, FIXEDWINDOW, MULTISCALE, LAB);

//txt_Writer
ofstream txtWriter("../dataRecord.txt",ios::out);

//FileStorage
FileStorage fs;

// first frame's target position
float xMin =  0;
float yMin =  0;
float width = 0;
float height = 0;
Point origin;
Rect TargetRect(xMin, yMin, width, height);

//Hand's TargetRect detected by CascadeClassifier
std::vector<Rect> hands;

//VideoCapture
VideoCapture cap(-1);

//knn_character_recognition
KNN_Character_Recognition KNN;

//svm_character_recognition
SVM_Character_Recognition SVM;

string outputNum;

// Frame readed
Mat frame;
Mat ROI;
Mat ROI_gray;
Mat sample;
Mat paint;
Mat background;

// Tracker results
Rect result;

void onMouse( int event, int x, int y, int, void* )
{
	static Mat temp;  //绘制选择框时用到
    if( selectObject )
    {

        TargetRect.x = MIN(x, origin.x);
        TargetRect.y = MIN(y, origin.y);
        TargetRect.width = std::abs(x - origin.x);
        TargetRect.height = std::abs(y - origin.y);

        TargetRect &= Rect(0, 0, frame.cols, frame.rows);

		frame.setTo(0);
		temp.copyTo(frame);
		rectangle(frame,origin,Point(x,y),Scalar(0,0,255),2);
		imshow("Image",frame);
    }

    switch( event )
    {
    case CV_EVENT_LBUTTONDOWN:
        origin = Point(x,y);
        TargetRect = Rect(x,y,0,0);
        selectObject = true;
		PAUSE = true;
		temp = frame.clone();
        break;
    case CV_EVENT_LBUTTONUP:
        selectObject = false;
		tracker.init( TargetRect, frame );
		temp.release();
		sample = frame(TargetRect);
		cvtColor(sample,sample,COLOR_BGR2GRAY);
		PAUSE = false;
		trackObject = true;
        break;
    }
}

#define _PYTHON_
#ifdef _PYTHON_

Python_TF_MINST py;

int main(int argc,char **argv){
    namedWindow( "Image", 0 );
    setMouseCallback( "Image", onMouse, 0 );
    namedWindow( "Paint", 0);

    // Read each frame from the cap
        if(!cap.read(frame))
            return -1;

    paint = Mat::zeros(frame.rows,frame.cols,CV_8UC3);
    background = Mat::zeros(frame.rows,frame.cols,CV_8UC3);

    py.Python_Init();
    PySys_SetArgv(argc,argv);
    py.Python_LoadModuleAndFunction();
    //py.test();
    while (1)
    {

        if(!PAUSE)
        {
        // Read each frame from the cap
        cap.read(frame);
        flip(frame,frame,1);
        // Update
        if(trackObject){
            result = tracker.update(frame);
            result &= Rect(0, 0, frame.cols, frame.rows);
            MouseProcess(result,tracker.flag);
            }
        imshow("Image", frame);
        }

            char c = waitKey(1);
                        switch(c)
                        {
                        case 'q':
                            return -1;
                        case 'p':
                            PAUSE=!PAUSE;
                            break;
                        case 'c':
                            trackObject=false;
                            tracker.flag=0;
                            break;
                        case 'b':
                            paint.setTo(0);
                            imshow("Paint",paint);
                            break;
                        case 's':
                            tracker.flag=!tracker.flag;
                            break;
                        case 'r':
                            color_num++;
                            if(color_num>2)color_num=0;
                            switch(color_num)
                            {
                            case 0:
                                color = Scalar(0,0,255); //红
                                break;
                            case 1:
                                color = Scalar(255,0,0); //蓝
                                break;
                            case 2:
                                color = Scalar(0,255,0); //绿
                                break;
                            }
                            break;
                        case 'o':
                            py.Opencv_imagePrepare(paint);
                            break;
                        }

       }
    py.Python_Final();
}
#else
int main(){
	namedWindow( "Image", 0 );
    setMouseCallback( "Image", onMouse, 0 );
    namedWindow( "Paint", 0);

	// Read each frame from the cap
		if(!cap.read(frame))
			return -1;

    KNN.readDataSet();
    KNN.train();
    SVM.readDataSet();
    SVM.train();

	paint = Mat::zeros(frame.rows,frame.cols,CV_8UC3);
	background = Mat::zeros(frame.rows,frame.cols,CV_8UC3);

	while (1)
	{

		if(!PAUSE)
		{
		// Read each frame from the cap
		cap.read(frame);
        flip(frame,frame,1);

		// Update
		if(trackObject){
			result = tracker.update(frame);
			result &= Rect(0, 0, frame.cols, frame.rows);
			MouseProcess(result,tracker.flag);
            }
		imshow("Image", frame);
		}

			char c = waitKey(1);
						switch(c)
						{
						case 'q':
							return -1;
						case 'p':
							PAUSE=!PAUSE;
							break;
						case 'c':
							trackObject=false;
							tracker.flag=0;
							break;
						case 'b':
							paint.setTo(0);
                            imshow("Paint",paint);
							break;
						case 's':
							tracker.flag=!tracker.flag;
                            if(tracker.flag == false)
                                KNN.classify(paint,outputNum);
							break;
						case 'r':
							color_num++;
							if(color_num>2)color_num=0;
							switch(color_num)
							{
							case 0:
								color = Scalar(0,0,255); //红
								break;
							case 1:
								color = Scalar(255,0,0); //蓝
								break;
							case 2:
								color = Scalar(0,255,0); //绿
								break;
							}
							break;
                        case 'o':
                            //KNN.classify(paint,outputNum);
                            SVM.classify(paint,outputNum);
							break;
						}

       }
}
#endif

void MouseProcess(Rect& trackBox,bool Flag)
{

	Point2f center = Point2f(trackBox.x+trackBox.width/2,trackBox.y+trackBox.height/2);
	static Point2f last_point = center;
/*
			//float pre_x=-1,pre_y=-1;//用于防抖
			//float scr_x=center.x,
			//	  scr_y=center.y,
			//	  delt;
			//int move_low=40,move_high=50;
			//bool flag=0;

			//if( (pre_x==-1) && (pre_y==-1) )
			//{
			//	pre_x=scr_x;
			//	pre_y=scr_y;
			//}
			//else 
			//{
			//	delt = sqrt( (scr_x-pre_x)*(scr_x-pre_x)+(scr_y-pre_y)*(scr_y-pre_y) );
			//	if( ( delt>move_low) && (delt<move_high) ) //SetCursorPos(scr_x,scr_y);
			//	pre_x=scr_x;
			//	pre_y=scr_y;
			//}
*/
		if(Flag)
		{
		rectangle( frame,result,Scalar( 0, 255, 0 ), 1, 8 );

		//由于计算机运算速度慢导致绘图时有不连续点出现，因此采用前后两点连线的方式解决了这个问题
		//太机智了！！！！
        circle(paint,center,THICKNESS,color,-1);
        line(paint,center,last_point,color,THICKNESS);

        imshow("Paint",paint);
		}
		else
		{
        circle(background,center,THICKNESS,color,-1);
		background |= paint;  //或运算,保证之前画的图像还在显示
        imshow("Paint",background);
		background.setTo(0);

		rectangle( frame,result,Scalar( 0, 255, 255 ), 1, 8 );
		}
		last_point = center;

}
