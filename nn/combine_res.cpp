#include <iostream>
#include <string>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <vector>


std::vector<std::string> SplitString(std::string s){
    std::vector<std::string> res;
    std::stringstream ss(s);
    std::string line;
    while(getline(ss, line,' ')){
        res.push_back(line);
    }
    return res;
}

void UpdateHashTable(std::unordered_map<std::string, std::pair<double, std::string> >& my_hash_table, std::string& line, const std::string& label ){
    std::vector<std::string> string_list = SplitString(line);
    if (string_list.size() != 2) {std::cout<<"data is wrong"<<std::endl;return;}

    double val = stod(string_list[1]);
    if (my_hash_table.count(string_list[0]) == 0){
        my_hash_table[string_list[0]] = std::make_pair(val, label);
    }
    else{
        if (my_hash_table[string_list[0]].first < val){
            my_hash_table[string_list[0]] = std::make_pair(val, label);
        }
    }
}


void PrintHashTable(std::unordered_map<std::string, std::pair<double, std::string> >&my_hash_table){
    for (auto item : my_hash_table){
        //std::cout<<item.first<<std::endl;
        std::cout<<item.second.first<<std::endl;
        std::cout<<item.second.second<<std::endl;
    }
}


int main(int argc, char* argv[]){
    std::string nba_dunk_filename = "../test_res_nba_dunk.txt";
    std::string nba_jumpshot_filename = "../test_res_nba_jumpshot.txt";
    std::string nba_layup_filename = "test_res_nba_layup.txt";
    
    std::ifstream nba_dunk_f(nba_dunk_filename);
    std::ifstream nba_jumpshot_f(nba_jumpshot_filename);
    std::ifstream nba_layup_f(nba_layup_filename);

    std::unordered_map<std::string, std::pair<double, std::string> > my_hash_table;

    std::string line;
    
    while(getline(nba_dunk_f, line)){
        UpdateHashTable(my_hash_table, line, "dunk");
    }
    while(getline(nba_jumpshot_f, line)){
        UpdateHashTable(my_hash_table, line, "jumpshot");
    }
    while(getline(nba_layup_f, line)){
        UpdateHashTable(my_hash_table, line, "layup");
    }
    PrintHashTable(my_hash_table);
}

