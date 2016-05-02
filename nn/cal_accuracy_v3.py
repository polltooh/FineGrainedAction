#! /usr/bin/env python
import utility_function as uf
import sys
import cv2
import numpy as np

def read_file_list(file_name):
    # assume the first argument is the image name and the second one is the label
    name_list = list()
    label_list = list()
    with open(file_name, "r") as f:
        s = f.read()
        s = uf.delete_last_empty_line(s)
        s_l = s.split("\n") 
        for ss in s_l:
            ss_l = ss.split(" ")
            assert(len(ss_l) == 2)
            name_list.append(ss_l[0])
            label_list.append(int((ss_l[1])))
    return name_list, label_list

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
    if (len(sys.argv) < 3):
        print("Usage: cal_accuracy_v3.py res_file_name.txt label_file.txt")
        exit(1)

    res_file_name = sys.argv[1]
    label_file_name =  sys.argv[2]

    res_name, res_list = read_file_list(res_file_name)

    label_name, label_list = read_file_list(label_file_name)
    diff_count = 0
    for i in range(len(res_name)):
        if (res_name[i] != label_name[i]):
            print("n1 is %s n2 is %s"%(n1,n2))
            exit(1)
    ave_precision = uf.cal_ave_precision(label_list, res_list, 12)
    con_mat = uf.cal_confusion_matrix(label_list, res_list, 12)
    np.save("ave_precision_v3.npy", ave_precision)
    np.save("con_mat_v3.npy", con_mat)
    print(uf.cal_ave_precision(label_list, res_list, 12))
    print(uf.cal_confusion_matrix(label_list, res_list, 12))
