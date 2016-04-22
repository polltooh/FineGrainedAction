#! /usr/bin/env python
import sys
import cv2

def delete_last_empty_line(s):
    while(len(s) >= 1 and s[-1] == '\n'):
        s = s[:-1]
    return s

def read_file(file_name):
    with open(file_name, "r") as f:
        file_data = f.read()
        file_data = delete_last_empty_line(file_data)
        data_list = file_data.split("\n")
        if (len(data_list) == 0):
            print("empty file " + file_name)
            return
        count = 0
        for i in range(len(data_list)):
            d_l = data_list[i].split(" ")
            if (int(float(d_l[1])) < 5):
                # print(d_l[1])
                image = cv2.imread(d_l[0])
                # cv2.imwrite("temp/%08d.jpg"%(count), image)
                count = count + 1
                cv2.imshow("res", image)
                cv2.waitKey(100)

if __name__ == "__main__":
    if (len(sys.argv) < 2):
        print("Usage: play_res.py res_file_name.txt")
        file_name = 'test_res_nba_dunk.txt'
    else:
        file_name = sys.argv[1]

    read_file(file_name)
