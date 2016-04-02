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
    std::cout<<boost::filesystem::exists(image_dir)<<std::endl;

    std::cout<<image_dir_name<<std::endl;
    std::string file_list_name = image_dir_name + "file_list.txt";
    std::system("ls " + image_dir_name + " > " + file_list_name);

    bool image_exist = utility::FileExist(file_list_name);

    // if (!image_exist){
    //     std::ofstream output_file(file_list_name.c_str());
    //     cv::VideoCapture cap(video_name);
    //     if(!cap.isOpened()){
    //         std::cout<<"video file is not found"<<std::endl;
    //         return -1;
    //     }

    //     std::string command = "mkdir -p " + image_dir_name;
    //     system(command.c_str());
    //     int index = 0;
    //     while(1){
    //         cv::Mat frame;
    //         cap >> frame;
    //         if(frame.empty()) break;
    //         char buffer[30];
    //         sprintf(buffer, "%08d.jpg",index);
    //         std::string bfs(buffer);
    //         std::string f_name = image_dir_name + bfs;
    //         std::cout<<f_name<<std::endl;
    //         cv::imwrite(f_name,frame);
    //         output_file << bfs <<'\n';
    //         index++;
    //     }
    // }
    // else{
    //     std::string file_list_name = image_dir_name + "file_list.txt";
    //     std::ifstream input_file(file_list_name);
    //     std::string line;
    //     std::vector<std::string> name_list;
    //     while(getline(input_file, line)){
    //         name_list.push_back(line);
    //     }
    //     std::vector<bool> label_list(name_list.size(), false);
    //     std::string label_list_name = image_dir_name + "label.txt";
    //     std::string label_list_cp_name = image_dir_name + "label_cp.txt";

    //     if (utility::FileExist(label_list_name)){
    //         std::ifstream label_file(label_list_name.c_str());
    //         int count = 0;
    //         while(getline(label_file, line)){
    //             label_list[count] = (line == "true");
    //             count++;
    //         }
    //     }

    //     int index = 0;
    //     bool stop = false;
    //     bool label = false;
    //     bool if_rolling = false;
    //     bool roll_direction = true;//right
    //     bool update_list = false;
    //     int move_factor = 1;
    //     while(1){
    //         if(stop) break;
    //         index = std::max(0, index);
    //         index = std::min((int)name_list.size() - 1, index);

    //         std::string image_file_name = image_dir_name + name_list[index];
    //         cv::Mat image = cv::imread(image_file_name);
    //         utility::AddText(image, label_name, label_list[index]);
    //         utility::AddFrameText(image, index);
    //         
    //         cv::imshow(label_name.c_str(), image);
    //         int key = 0;
    //         if (if_rolling) key = cv::waitKey(20);
    //         if (!if_rolling) key = cv::waitKey(0);
    //         utility::KeyBehavior(key, label,index, if_rolling,
    //             roll_direction, update_list, stop, move_factor);
    //         if (update_list) utility::UpdateLabel(label_list, label, index);
    //         if (index != 0 && (index % 1000 == 1)) 
    //             utility::WriteToTxt(label_list_cp_name, label_list);
    //         //if (if_rolling) index += int(roll_direction) * 2 - 1;
    //     }
    //     utility::WriteToTxt(label_list_name, label_list);
    // }
    
    return 0;
}

