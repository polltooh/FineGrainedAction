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
#include "img_uti.h"

int main(int argc, char* argv[]){

    if (argc < 2){
        std::cout<<"Usage: delete_non_image.bin image_folder_name"<<std::endl;
        return 0;
    }
    std::string image_dir = argv[1];
     
    if (!utility::FileExist(image_dir)){
        std::cout<< image_dir + " does not exist"<<std::endl;
        return 0;
    }

    std::vector<std::string> file_list = utility::FileInDir(image_dir);
    std::vector<std::string> remove_file_list;
    for (size_t i = 0; i < file_list.size(); ++i){
        if (!utility::StrEndWith(file_list[i], "jpg") 
           && !utility::StrEndWith(file_list[i],"png") 
           && !utility::StrEndWith(file_list[i],"jpeg")){
                remove_file_list.push_back(file_list[i]);
                std::cout<<file_list[i] + " will be deleted"<< std::endl;
        }
    }

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

    if (image_dir[image_dir.size()-1] != '/'){
        image_dir += '/';
    }
    for (size_t i = 0; i < remove_file_list.size(); ++i){
        std::cout<<remove_file_list[i] + " is deleted"<< std::endl;
        std::remove(remove_file_list[i].c_str());
    }
    

    file_list = utility::FileInDir(image_dir);
    int count = 0;
    for (int i = 0; i < file_list.size(); ++i){
        cv::Mat image = cv::imread(file_list[i]);
        std::remove(file_list[i].c_str());
        if (!image.data) continue;

        cv::Mat resized_image = img_uti::CropResize(image, 256);
        char buffer [50];
        sprintf (buffer, "%08d.jpg", count);
        std::string new_image_name = image_dir + std::string(buffer);
        count++;
        cv::imwrite(new_image_name, resized_image);
    }
}
