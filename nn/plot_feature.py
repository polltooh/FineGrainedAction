import numpy as np
from sklearn.decomposition import PCA
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

image_dir = "../data/image/"
frame_dir = "../data/test_video/"

def delete_last_empty_line(s):
    while(len(s) >= 1 and s[-1] == '\n'):
        s = s[:-1]
    return s

def get_image(label_name):
    list_name = os.listdir(image_dir + label_name)
    list_name = [f for f in list_name if f[0:2] != '._' and f.endswith('.jpg')]
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
    return image

def get_list_2(label_name):
    pos_frame, neg_frame = get_frame(label_name)
    return pos_frame

def read_feature(label_name):
    # image_list = get_list(label_name)
    image_list = get_list_2(label_name)
    feature_list = list()    
    for i in range(len(image_list)):
        feature_name = image_list[i].replace(".jpg",".tri_fc7")
        feature = np.fromfile(feature_name, np.float32)
        # feature_name = image_list[i].replace(".jpg",".cn5")
        # feature_2 = np.fromfile(feature_name, np.float32)
        # feature = np.hstack((feature_1, feature_2))
        feature_list.append(feature)
    feature_np = np.array(feature_list)
    return feature_np

def cal_pca(feature_np_list):
    pca = PCA(n_components=3)
    feature_np = feature_np_list[0]
    range_list = [len(feature_np_list[0])]
    for i in range(len(feature_np_list))[1:]:
        feature_np = np.vstack((feature_np, feature_np_list[i]))
        range_list += [range_list[-1] + len(feature_np_list[i])]

    pca_feat = pca.fit_transform(feature_np)
    return pca_feat, range_list

def plot_pca(pca_feat,range_list):
    # color_list = ["r","r","r", "b", "b","b", "g", "g", "g"]
    # color_list = ["r", "r", "r", "b", "b", "b"]
    color_list = ["r", "b", "g"]
    range_list.insert(0, 0)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # plt.xlim(-100,100)
    # plt.ylim(-100,100)
    for i in range(len(range_list))[:-1]:
        # ax.scatter(pca_feat[range_list[i]:range_list[i+1],0], \
        #         pca_feat[range_list[i]:range_list[i+1],1], \
        #         pca_feat[range_list[i]:range_list[i+1],2], \
        #         c = color_list[i])
        plt.plot(pca_feat[range_list[i]:range_list[i+1],0], \
            pca_feat[range_list[i]:range_list[i+1],1],color_list[i] + "o")
    plt.show()

if __name__ == "__main__":
    pca_feat_list = list()
    feature_np_list = list()
    feature_np_list.append(read_feature("nba_dunk"))
    feature_np_list.append(read_feature("nba_jumpshot"))
    feature_np_list.append(read_feature("nba_layup"))

    # feature_np_list.append(read_feature("tennis_forehand"))
    # feature_np_list.append(read_feature("tennis_backhand"))
    # feature_np_list.append(read_feature("tennis_serve"))

    # feature_np_list.append(read_feature("baseball_hit"))
    # feature_np_list.append(read_feature("baseball_stolen_base"))
    # feature_np_list.append(read_feature("baseball_pitch"))

    pca_feat, range_list = cal_pca(feature_np_list)
    plot_pca(pca_feat, range_list)
