#ifndef UTILITY_FUNCTION_H_
#define UTILITY_FUNCTION_H_
#include <vector>
#include <fstream>
#include <algorithm>
#include <iostream>
#include <string.h>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>

const int LEFT_KEY = 63234;
const int RIGHT_KEY = 63235;
const int UP_KEY = 63232;
const int DOWN_KEY = 63233;

namespace utility{

// for gui
void UpdateLabel(std::vector<bool>& label_list, bool label, int index);
void AddText(cv::Mat& image, std::string label_name, bool label);
void AddFrameText(cv::Mat& image, int index);

// for saing the label
template<typename T>
void WriteToTxt(std::string file_name, std::vector<T> list);

template<>
void WriteToTxt(std::string file_name, std::vector<bool> list);

void KeyBehavior(int& key, bool& label, int& index,bool& if_rolling, 
    bool& roll_direction, bool& update_list, bool& stop, int& move_factor);

void KeyBehavior(int& key, bool& label, int& index, bool& updata_label, bool& stop);

std::vector<std::string> SplitString(std::string video_name);

bool FileExist(std::string file_name);
std::vector<std::string> FileInDir(std::string dir_name);

inline bool StrEndWith(std::string filename, std::string extension);

}
#endif
