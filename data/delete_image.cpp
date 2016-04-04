#include <stdio.h>

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
    int delete_count = 0;
    for (int i = 0; i < file_list.size(); ++i){
        if (label_list[i] == false){
            delete_count ++;
            std::remove(file_list[i].c_str());
        }
    }
    std::cout<<delete_count<<std::endl;

}
