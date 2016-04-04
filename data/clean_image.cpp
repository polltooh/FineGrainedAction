#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <string>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <stdlib.h>
#include "boost/filesystem.hpp"
#include "utility_function.h"

int main(int argc, char* argv[]){
    if (argc < 2){
        std::cout<<"Usage: CleanImage image_folder_name"<<std::endl;
        return 0;
    }
    std::string image_dir = argv[1];
     
    if (!utility::FileExist(image_dir)){
        std::cout<< image_dir + " does not exist"<<std::endl;
        return 0;
    }

    if (image_dir[image_dir.size()-1] != '/'){
        image_dir += '/';
    }
        
    std::vector<std::string> split_name = utility::SplitString(image_dir); 

    std::vector<std::string> file_list = utility::FileInDir(image_dir);
    
    std::string label_list_name = image_dir + "label.txt";
    std::string label_list_cp_name = image_dir + "label_cp.txt";

    std::vector<bool> label_list(file_list.size(), true);

    if(utility::FileExist(label_list_name)){
        std::ifstream label_file(label_list_name.c_str());
        std::string line;
        int count = 0;
        while(getline(label_file, line)){
            label_list[count] = (line == "true");
            count++;
        }
    }

    std::string label_name = split_name[split_name.size()-1];
    bool label = true;
    bool stop = false;
    bool update_label = false;
    int index = 0;
    while(1){
        if(stop) break;
        index = std::max(0, index);
        index = std::min((int)label_list.size() - 1, index);
        label = label_list[index];
        cv::Mat image = cv::imread(file_list[index]);
        cv::Mat resized_image;

        if (!image.data){
           resized_image = cv::Mat::zeros(400, 400, CV_32F);
        }
        else{
            cv::resize(image, resized_image, cv::Size(400,400));
        }

        utility::AddText(resized_image, label_name, label_list[index]);
        utility::AddFrameText(resized_image, index);

        cv::imshow("test", resized_image);
        int key = cv::waitKey(0);
        utility::KeyBehavior(key, label, index, update_label, stop);

        if (update_label && index >= 0 && index < label_list.size()){
            label_list[index] = label;
        }
        
        if (index != 0 && (index % 50 == 1)) 
            utility::WriteToTxt(label_list_cp_name, label_list);
    }
    
    utility::WriteToTxt(label_list_name, label_list);

    return 0;
}

