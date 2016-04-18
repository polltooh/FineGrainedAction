import os
import cv2
import random
import numpy as np

image_dir = "/home/mscvadmin/action/FineGrainedAction/data/image/"
frame_dir = "/home/mscvadmin/action/FineGrainedAction/data/video/"

def delete_last_empty_line(s):
    while(len(s) >= 1 and s[-1] == '\n'):
        s = s[:-1]
    return s

def get_image(label_name):
    list_name = os.listdir(image_dir + label_name)
    list_name = [f for f in list_name if f[0:2] != '._']
    list_name = [image_dir + label_name + "/" + f for f in list_name]
    return list_name

def get_frame(label_name):
    list_name = os.listdir(frame_dir + label_name)
    pos_list = list()
    neg_list = list()
    for name in list_name:
       curr_path = frame_dir + label_name + "/" + name + "/"
       if (os.path.isdir(curr_path)):
           file_list_name = curr_path + "/file_list.txt"
           label_list_name = curr_path + "/label.txt"            
           with open (file_list_name, "r") as f:
               file_data = (f.read())
               file_data = delete_last_empty_line(file_data)
               data_list = file_data.split("\n")
               # print(data_list)
           with open (label_list_name, "r") as f:
               label_data = f.read()
               label_data = delete_last_empty_line(label_data)
               label_list = label_data.split("\n")
           for index in range(len(label_list)):
               if (label_list[index] == "true"):
                   pos_list.append(curr_path + data_list[index])
               else:
                   neg_list.append(curr_path + data_list[index])
    return pos_list, neg_list

def get_list(label_name):
    image = get_image(label_name)
    frame_pos, frame_neg = get_frame(label_name)
    return image, frame_pos, frame_neg 

def gen_list_2(file_name_list):
    # file name 1: the positive image label
    # file name 2 & 3..: the negative image label
    file_num = len(file_name_list)
    image_d = dict()
    frame_pos_d = dict()
    frame_neg_d = dict()
    for i in range(file_num):
        image_d[i], frame_pos_d[i], frame_neg_d[i] = get_list(file_name_list[i])


    neg_full = frame_neg_d[0]
    for i in range(file_num):
        if i == 0:
            continue
        neg_full += frame_pos_d[i]

    random.seed(10)

    random.shuffle(neg_full)
    time_num = 10
    pos_index = range(len(frame_pos_d[0]))
    pos_index = pos_index * time_num
    pos_index = random.sample(pos_index, time_num * len(image_d[0]))
    np_pos_index = np.reshape(np.array(pos_index), (len(image_d[0]), time_num))
    
    neg_index = range(len(neg_full))
    neg_index = neg_index * time_num
    neg_index = random.sample(neg_index, time_num * len(image_d[0]))
    np_neg_index = np.reshape(np.array(neg_index), (len(image_d[0]), time_num))
    
    file_list = list()
    for i in range(len(image_d[0])):
        for j in range(time_num):
            curr_s = image_d[0][i] + " " + frame_pos_d[0][np_pos_index[i][j]] + \
                " " + "1" + "\n"
            file_list.append(curr_s)
            curr_s = image_d[0][i] + " " + neg_full[np_neg_index[i][j]] + \
                " " + "0" + "\n"
            file_list.append(curr_s)
            # file_list.append(" ")
            # f.write(image_d[0][i])
            # f.write(" ")
            # f.write(frame_pos_d[0][np_pos_index[i][j]])
            # f.write(" ")
            # f.write("1")
            # f.write("\n")

            # f.write(image_d[0][i])
            # f.write(" ")
            # f.write(neg_full[np_neg_index[i][j]])
            # f.write(" ")
            # f.write("0")
            # f.write("\n")
    return file_list

def gen_list():
    dunk_image, dunk_frame_pos, dunk_frame_neg = get_list("nba_dunk")
    jumpshot_image, jumpshot_frame_pos, jumpshot_frame_neg = get_list("nba_jumpshot")
    layup_image, layup_frame_pos, layup_frame_neg = get_list("nba_layup")
    file_name = "file_list_train.txt"
    dunk_neg_full = dunk_frame_neg + jumpshot_frame_pos + layup_frame_pos
    random.seed(10)

    random.shuffle(dunk_neg_full)
    time_num = 100
    pos_index = range(len(dunk_frame_pos))
    pos_index = pos_index * time_num
    pos_index = random.sample(pos_index, time_num * len(dunk_image))
    np_pos_index = np.reshape(np.array(pos_index), (len(dunk_image), time_num))

    
    neg_index = range(len(dunk_neg_full))
    neg_index = neg_index * time_num
    neg_index = random.sample(neg_index, time_num * len(dunk_image))
    np_neg_index = np.reshape(np.array(neg_index), (len(dunk_image), time_num))
    
    with open(file_name,  "w") as f:
        for i in range(len(dunk_image)):
            for j in range(time_num):
                f.write(dunk_image[i])
                f.write(" ")
                f.write(dunk_frame_pos[np_pos_index[i][j]])
                f.write(" ")
                f.write("1")
                f.write("\n")

                f.write(dunk_image[i])
                f.write(" ")
                f.write(dunk_neg_full[np_neg_index[i][j]])
                f.write(" ")
                f.write("0")
                f.write("\n")



if __name__ == "__main__":
    curr_list = list()
    curr_list += gen_list_2(['nba_dunk', 'nba_jumpshot', 'nba_layup'])
    curr_list += gen_list_2(['nba_jumpshot', 'nba_dunk', 'nba_layup'])
    curr_list += gen_list_2(['nba_layup', 'nba_jumpshot', 'nba_dunk'])

    with open("file_list_train.txt", "w") as f:
        for c in curr_list:
            f.write(c)

