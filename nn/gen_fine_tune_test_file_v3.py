import os
import cv2
import random
import numpy as np

image_dir = "../data/test_image/"
frame_dir = "../data/test_video/"

label_dict = {
    'nba_neg':'0',
    'nba_dunk':'1',
    'nba_jumpshot':'2',
    'nba_layup':'3',
    'tennis_neg':'4',
    'tennis_forehand':'5',
    'tennis_backhand':'6',
    'tennis_serve':'7',
    'baseball_neg':'8',
    'baseball_hit':'9',
    'baseball_pitch':'10',
    'baseball_stolen_base':'11'
}

print(label_dict)
random.seed(10)

def delete_last_empty_line(s):
    while(len(s) >= 1 and s[-1] == '\n'):
        s = s[:-1]
    return s

def get_image(label_name):
    list_name = os.listdir(image_dir + label_name)
    list_name = [f for f in list_name if f[0:2] != '._']
    list_name = [f for f in list_name if f.endswith(".jpg")]
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

def get_neg_name(label_name):
    neg_tag = label_name.split('_')
    neg_name = neg_tag[0] + "_neg"
    return neg_name


def get_list(label_name, suffle_cut_neg = False):
    """if suffle_cut_neg set to be True, it will cut the negative 
        sample to be the number of possitive sample"""
    image = get_image(label_name)
    frame_pos, frame_neg = get_frame(label_name)

    if suffle_cut_neg:
        random.shuffle(frame_neg)
        frame_neg = frame_neg[0:min(len(frame_neg), len(image) + len(frame_pos))]

    return image, frame_pos, frame_neg 

def write_to_file(full_image_list, label_list, file_name):
    index = range(len(full_image_list))
    random.shuffle(index)

    with open(file_name,  "w") as f:
        for i in range(len(full_image_list)):
            f.write(full_image_list[index[i]])
            f.write(" ")
            f.write(str(label_list[index[i]]))
            f.write("\n")


def gen_list(file_name_list, pos_start_count):
    file_num = len(file_name_list)
    name_list = list()
    label_list = list()
    
    neg_label = label_dict[get_neg_name(file_name_list[0])]

    for i in range(file_num):
        pos_label = label_dict[file_name_list[i]]
        _, frame_pos_l, frame_neg_l = get_list(file_name_list[i], False)
        name_list += frame_pos_l + frame_neg_l
        label_list += [pos_label] * (\
                len(frame_pos_l)) + [neg_label] * len(frame_neg_l)

    return name_list, label_list 

if __name__ == "__main__":
    
    name_list = list()
    label_list = list()

    name_list_t, label_list_t = gen_list(['nba_dunk', 'nba_jumpshot', \
                'nba_layup'], 1)
    name_list += name_list_t
    label_list += label_list_t

    name_list_t, label_list_t = gen_list(['tennis_forehand', 'tennis_backhand', \
                'tennis_serve'], 4)
    name_list += name_list_t
    label_list += label_list_t

    name_list_t, label_list_t = gen_list(['baseball_hit', 'baseball_pitch', \
                'baseball_stolen_base'], 7)
    name_list += name_list_t
    label_list += label_list_t

    file_name = "file_list_fine_tune_test_v3.txt"

    write_to_file(name_list, label_list, file_name)
