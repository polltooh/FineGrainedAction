#! /usr/bin/env python
import utility_function as uf
import sys
import cv2

# label_file = "../data/test_video/nba_dunk/the_top_10_nba_dunks_of_all_time/label.txt"

# label_file = "../data/test_video/nba_dunk/The_Top_10_NBA_Dunks_Of_All_Time/label.txt"

def read_label(label_file):
    with open(label_file, "r") as f:
        s = f.read();
        s = uf.delete_last_empty_line(s)
        s_l = s.split("\n")
        s_l = [s == "true" for s in s_l]
    return s_l

def read_fine_tune_res_file(file_name, res_list, label_num):
    with open(file_name, "r") as f:
        file_data = f.read()
        file_data = uf.delete_last_empty_line(file_data)
        data_list = file_data.split("\n")
        if (len(data_list) == 0):
            print("empty file " + file_name)
            return
        for i in range(len(data_list)):
            d_l = data_list[i].split(" ")
            if (int(float(d_l[1])) == label_num):
                index = uf.file_name_to_int(d_l[0])
                res_list[index] = True

def read_triplet_res_file(file_name, res_list, radius):
    with open(file_name, "r") as f:
        file_data = f.read()
        file_data = uf.delete_last_empty_line(file_data)
        data_list = file_data.split("\n")
        if (len(data_list) == 0):
            print("empty file " + file_name)
            return
        for i in range(len(data_list)):
            d_l = data_list[i].split(" ")
            if (float(d_l[1]) < radius):
                index = uf.file_name_to_int(d_l[0])
                res_list[index] = True

                # image = cv2.imread(d_l[0])
                # cv2.imshow("res", image)
                # cv2.waitKey(100)

if __name__ == "__main__":
    if (len(sys.argv) < 4):
        print("Usage: play_res.py res_file_name.txt label_file.txt loss_type[0/1] tun_param")
        # file_name = 'test_res_nba_dunk.txt'
        # label_file = "../data/test_video/nba_dunk/the_top_10_nba_dunks_of_all_time/label.txt"
        # loss_type = 0
        exit(1)

    file_name = sys.argv[1]
    label_file =  sys.argv[2]
    loss_type = sys.argv[3]
    tun_param = sys.argv[4]

    label = read_label(label_file)
    res_list = [False] * len(label)

    if loss_type == '0':
        read_triplet_res_file(file_name, res_list, float(tun_param))
    else:
        read_fine_tune_res_file(file_name, res_list, int(float(tun_param)))

    print("precision is: %f" % (uf.cal_percision(label, res_list)) )
    print("recall is: %f" % (uf.cal_recall(label, res_list)) )
    # read_file(file_name)
