#include "utility_function.h"

void utility::UpdateLabel(std::vector<bool>& label_list, bool label, int index){
   while(index < label_list.size() && label_list[index] != label){
        label_list[index] = label;
        index ++;
   }
}

void utility::AddText(cv::Mat& image, std::string label_name, bool label){
    std::string text = label ? label_name : "negative";
    const int col_offset = 100;
    const int row_offset = 20;
    cv::Point position((image.cols)/2 - col_offset, image.rows - row_offset);
    cv::Scalar color(255 * int(label), 255, 0); 
    cv::putText(image, text , position, cv::FONT_HERSHEY_SIMPLEX, 1, color, 2);
}

void utility::AddFrameText(cv::Mat& image, int index){
    std::string text = std::to_string(index);
    const int col_offset = 100;
    const int row_offset = 20;
    cv::Point position((image.cols - col_offset)/2,row_offset);
    cv::Scalar color(255, 255, 0); 
    cv::putText(image, text , position, cv::FONT_HERSHEY_SIMPLEX, 1, 
        color, 2);

}

template <typename T>
void utility::WriteToTxt(std::string file_name, std::vector<T> list){

    std::ofstream output_file((file_name).c_str());

    for (int i = 0; i < list.size(); ++i){
        output_file << list[i]<<'\n';
    }
}

template <>
void utility::WriteToTxt(std::string file_name, std::vector<bool> list){

    std::ofstream output_file((file_name).c_str());

    for (int i = 0; i < list.size(); ++i){
        output_file << std::string(list[i] ? "true": "false")<<'\n';
    }
}

void utility::KeyBehavior(int& key, bool& label, int& index, 
    bool& if_rolling, bool& roll_direction, bool& update_list, 
    bool& stop, int& move_factor){

    update_list = false;
    switch (char(key)){
        case 'a':
            if_rolling = false; label = true; update_list = true;
            break;
        case 's':
            if_rolling = false; label = false; update_list = true;
            break;
        case char(RIGHT_KEY):
            index++; if_rolling = false;
            break;
        case char(LEFT_KEY):
            index--; if_rolling = false;
            break;
        case char(UP_KEY):
            if_rolling = true; roll_direction = true;
            break;
        case char(DOWN_KEY):
            if_rolling = true; roll_direction = false;
            break;
        case 'q':
            stop = true;
            break;
        case '=':
            if(if_rolling){
                move_factor *= 2;
            }
            break;
        case '-':
            if (if_rolling){
                move_factor /= 2;
                move_factor = std::max(1, move_factor);
            }
            break;
    }
    if(if_rolling){
        if (roll_direction) index += move_factor;        
        if (!roll_direction) index -= move_factor;        
    }
    else{
        move_factor = 1;
    }
}

void utility::KeyBehavior(int& key, bool& label, int& index, 
    bool& update_label, bool& stop){

    update_label = false;

    switch (char(key)){
        case 'a':
            label = true; update_label = true;
            break;
        case 's':
            label = false; update_label = true;
            break;
        case char(RIGHT_KEY):
            index++;
            break;
        case char(LEFT_KEY):
            index--;
            break;
        case 'q':
            stop = true;
            break;
    }
}

std::vector<std::string> utility::SplitString(std::string video_name){
    std::vector<std::string> name_list;
    char buffer[200] = {0};
    strcpy(buffer, video_name.c_str());
    char* pch = std::strtok(buffer, "./\\");
    while(pch != nullptr){
        name_list.push_back(pch);
        pch = std::strtok(nullptr,"./\\");
    }
    return name_list;
}


bool utility::FileExist(std::string file_name){
    return boost::filesystem::exists(file_name);
}

std::vector<std::string> utility::FileInDir(std::string dir_name){
    std::vector<std::string> file_name_list;
    boost::filesystem::path p(dir_name);
    boost::filesystem::directory_iterator end_itr;
    for (boost::filesystem::directory_iterator itr(p); itr != end_itr; ++itr){
        if (is_regular_file(itr->path())) {
            std::string current_file = itr->path().string();
            if (StrEndWith(current_file, "jpg") || StrEndWith(current_file, "png")
                ||StrEndWith(current_file, "jpeg"))

                file_name_list.push_back(current_file);
            else
                std::cout<<current_file<<std::endl;
        }
    }
    return file_name_list;
}

inline bool utility::StrEndWith(std::string filename, std::string extension){
    if (filename.size() < extension.size()) return false;
    if (strcasecmp(filename.substr(filename.size() - extension.size()).c_str(), extension.c_str())) return true; 
    //if (filename.substr(filename.size() - extension.size()) == extension) return true;
    else return false;

}
