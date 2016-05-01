import sys
import os
import cv2
import utility_function as uf
import numpy as np

def parse_file(file_list_name):
    with open(file_list_name, "r") as f:
        s = f.read()
        s = uf.delete_last_empty_line(s)
        s_l = s.split("\n")
        for lname in s_l:
            list_name = lname.split(" ")
            if (len(list_name) == 3):
                img = read_pair_image(list_name)

                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img,str(list_name[2]),(10,500), font, 4,(255,255,255),2)

                cv2.imshow("image", img)
                cv2.waitKey(1000)

def read_pair_image(list_name):
    img1 = cv2.imread(list_name[0])
    img2 = cv2.imread(list_name[1])
    img = np.vstack((img1, img2))
    return img

def read_image(image_name):
    img = cv2.imread(image_name)
    return img

if __name__ == "__main__":
    if (len(sys.argv) < 2):
        print("Usage: play_file_list.py file_list_name.txt")
        exit(1)
    file_list_name = sys.argv[1]
    parse_file(file_list_name)
