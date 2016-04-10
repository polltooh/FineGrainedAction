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
        std::cout<<"Usage: delete_wrong_image.bin image_folder_name"<<std::endl;
        return 0;
    }
    std::string image_dir = argv[1];
     
    if (!utility::FileExist(image_dir)){
        std::cout<< image_dir + " does not exist"<<std::endl;
        return 0;
    }
    else{
        std::cout<<"Are you sure you want to delete filesd in " + image_dir << " [y/n]"<<std::endl;
        char sure = 'n';
        std::cin >> sure;
        if (sure == 'n') {
            std::cout<<"undeleted"<<std::endl;
            return 0;
        }
        else if(sure == 'y'){
            std::cout<<"deleting"<<std::endl;
        }
        else{
            std::cout<<"unknown command"<<std::endl;
            return 0;
        }
    }
    if (image_dir[image_dir.size()-1] != '/'){
        image_dir += '/';
    }
    
    std::string name_list_name = image_dir + "name_list.txt";

    std::vector<std::pair<std::string, bool>> name_list;
    std::vector<std::string> image_list;
    std::vector<bool> label_list;

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
        std::cout<<"this file have not been annotated yet"<<std::endl;
        return 0;
    }

    int delete_count = 0;
    for (int i = 0; i < image_list.size(); ++i){
        if (label_list[i] == false){
            delete_count ++;
            std::remove(image_list[i].c_str());
        }
    }

    std::cout<<delete_count<<std::endl;

    if(utility::FileExist(name_list_name)){
        std::remove(name_list_name.c_str());    
    }

}
