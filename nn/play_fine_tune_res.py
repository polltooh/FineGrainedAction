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
        count = 0
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i in range(len(data_list)):
            d_l = data_list[i].split(" ")
            if (d_l[1] == '1'):
                image = cv2.imread(d_l[0].replace(".tri_fc7",".jpg"))
                cv2.putText(image, d_l[1], (100,200), font, 4,(255,255,255),1)
                # cv2.imwrite("temp/%08d.jpg"%(count), image)
                count = count + 1
                cv2.imshow("res", image)
                cv2.waitKey(100)


if __name__ == "__main__":
    if (len(sys.argv) < 2):
        print("Usage: play_res.py res_file_name.txt")
        file_name = 'fine_tune_test_res.txt'
    else:
        file_name = sys.argv[1]

    read_file(file_name)
