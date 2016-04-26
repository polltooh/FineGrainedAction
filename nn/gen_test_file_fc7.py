import os
import cv2
import random
import numpy as np

image_dir = "/home/mscvadmin/action/FineGrainedAction/data/test_image/"
frame_dir = "/home/mscvadmin/action/FineGrainedAction/data/test_video/"

def delete_last_empty_line(s):
    while(len(s) >= 1 and s[-1] == '\n'):
        s = s[:-1]
    return s

# assume there is only one image
def get_image(label_name):
    list_name = os.listdir(image_dir + label_name)
    list_name = [image_dir + label_name + "/" + f.replace(".jpg", ".fc7") for f in list_name]
    return list_name[0]

def get_frame(label_name):
    list_name = os.listdir(frame_dir + label_name)
    frame_list = list()
    for name in list_name:
       curr_path = frame_dir + label_name + "/" + name + "/"
       if (os.path.isdir(curr_path)):
           file_list_name = curr_path + "/file_list.txt"
           with open (file_list_name, "r") as f:
               file_data = (f.read())
               file_data = delete_last_empty_line(file_data)
               data_list = file_data.split("\n")
               for d in data_list:
                   frame_list.append(curr_path + d.replace(".jpg", ".fc7"))

    return frame_list

def get_list(label_name):
    image = get_image(label_name)
    frame_list = get_frame(label_name)
    return image, frame_list

def gen_list():
    label_name = "nba_dunk"
    query_image, test_frame = get_list("nba_dunk")
    # query_image = get_image(label_name)
    # jumpshot_image = get_image("nba_jumpshot")
    # layup_image = get_image("nba_layup")
    with open("file_list_test_" + label_name + "_fc7.txt",  "w") as f:
        for i in range(len(test_frame)):
            f.write(query_image)
            f.write(" ")
            f.write(test_frame[i])
            f.write(" ")
            f.write("-1")
            f.write("\n")

            # f.write(jumpshot_image)
            # f.write(" ")
            # f.write(test_frame[i])
            # f.write(" ")
            # f.write("-1")
            # f.write("\n")

            # f.write(layup_image)
            # f.write(" ")
            # f.write(test_frame[i])
            # f.write(" ")
            # f.write("-1")
            # f.write("\n")

if __name__ == "__main__":
    gen_list()

