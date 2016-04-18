#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <stdlib.h>
#include <utility> 

#include "boost/filesystem.hpp"
#include "utility_function.h"

//template<typename T1, typename T2, typename T3>
//void insertToPair(std::vector<std::pair<T1, T2>>& name_list, std::vector<T3>& label, int index){
//    std::cout<<"here";
//}

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

    std::vector<std::pair<std::string, bool>> name_list;
    std::vector<bool> label_list;
    std::vector<std::string> image_list;

    std::string name_list_name = image_dir + "name_list.txt";

    if (utility::FileExist(name_list_name)){
        std::ifstream name_file(name_list_name.c_str());
        std::string line;
        while(getline(name_file, line)){
            std::istringstream line_list(line);
            std::string image_name, label_name;
            getline(line_list, image_name, ' ');
            getline(line_list, label_name, ' ');
            name_list.push_back(std::make_pair(image_name, label_name == "true"));
            image_list.push_back(image_name);
            label_list.push_back(label_name == "true");
        }
    }
    else{
        image_list = utility::ImageInDir(image_dir);  
        // for (int i = 0; i < image_list.size(); ++i){
        //     std::cout<<image_list[i]<<std::endl;
        // }
        name_list = std::vector<std::pair<std::string, bool>>(image_list.size()); 
        label_list = std::vector<bool>(image_list.size(), true);

        utility::InsertToPair(name_list, image_list);
        utility::InsertToPair(name_list, label_list);

        utility::WriteToTxt(name_list_name, name_list);
    }

    // std::string name_list_cp_name = image_dir + "label_cp.txt";

    // std::vector<bool> label_list(image_list.size(), true);

    /*
    if(utility::FileExist(name_list_name)){
        std::ifstream label_file(name_list_name.c_str());
        std::string line;
        int count = 0;
        while(getline(label_file, line)){
            std::string line_temp = utility::SplitString(line);
            label_list[count] = (line_temp[1] == "true");
            count++;
        }
    }
    
*/
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
        cv::Mat image = cv::imread(image_list[index]);
        cv::Mat resized_image;
        cv::resize(image, resized_image, cv::Size(image.rows * 2, image.cols * 2));

        utility::AddText(resized_image, label_name, label_list[index]);
        utility::AddFrameText(resized_image, index, int(label_list.size() - 1));

        cv::imshow("test", resized_image);
        int key = cv::waitKey(0);
        utility::KeyBehavior(key, label, index, update_label, stop);

        if (update_label && index >= 0 && index < label_list.size()){
            label_list[index] = label;
        }
        
        if (index != 0 && (index % 50 == 1)) 
            utility::InsertToPair(name_list, label_list);
            utility::WriteToTxt(name_list_name, name_list);
    }
    
    utility::InsertToPair(name_list, label_list);
    utility::WriteToTxt(name_list_name, name_list);
    return 0;
}

