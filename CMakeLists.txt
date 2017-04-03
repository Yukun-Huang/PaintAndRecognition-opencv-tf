#工程项目名称
PROJECT(handWriting)  			

CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

#选择编译器和c++标准(c++98 c++03 c++11)
SET(CMAKE_CXX_COMPILER "/usr/bin/g++")
add_compile_options(-std=c++11)

# debug模式

#CMake查找opencv包
FIND_PACKAGE( OpenCV REQUIRED )		

#CMake查找python包
FIND_PACKAGE(PythonLibs REQUIRED)
if(PYTHONLIBS_FOUND)
    MESSAGE(STATUS "PYTHONLIBS FOUND!")
#   MESSAGE(STATUS "PYTHON_LIBRARIES: ${PYTHON_LIBRARIES}")
#   MESSAGE(STATUS "PYTHON_INCLUDE_DIR: ${PYTHON_INCLUDE_DIR}")
endif()

# Declare the target (an executable)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_DIR})

ADD_EXECUTABLE(handWriting 
 src/main.cpp
 src/kcftracker.cpp
 src/fhog.cpp
 src/knn_character_recognition.cpp
 src/svm_character_recognition.cpp
 src/python_tf_minst.cpp)

TARGET_LINK_LIBRARIES(handWriting ${OpenCV_LIBS} ${PYTHON_LIBRARIES})

# MESSAGE(STATUS "PYTHON_LIBRARIES: ${PYTHON_LIBRARIES}")
# MESSAGE(STATUS "PYTHON_INCLUDE_DIR: ${PYTHON_INCLUDE_DIR}")

# MESSAGE(STATUS "CMAKE_BINARY_DIR: ${CMAKE_BINARY_DIR}")
# MESSAGE(STATUS "CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")

# 显示opencv库状态
# MESSAGE(STATUS "OpenCV_DIR: ${OpenCV_DIR}")
# MESSAGE(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")
