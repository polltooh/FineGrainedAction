import tensorflow as tf
from time import gmtime, strftime

def delete_last_empty_line(s):
    end_index = len(s) - 1
    while(end_index >= 0 and s[end_index] == '\n'):
        end_index -= 1
    s = s[:end_index + 1]
    return s

def file_name_to_int(file_name):
    # seperate the file name from the path.
    # assume the last one is the filename
    file_name = file_name.split("/")[-1]
    # in case the filename contains extension
    file_no_ext = file_name.split(".")[0]
    return int(file_no_ext)

def read_binary(filename, dim):
    bin_file = tf.read_file(filename)
    bin_tensor = tf.decode_raw(bin_file, tf.float32)
    # bin_tensor = tf.to_float(bin_tensor)
    bin_tensor = tf.reshape(bin_tensor,[dim])
    return bin_tensor


def cal_percision(label_list, classify_res):
    assert(len(label_list) == len(classify_res))
    true_positive_count = 0.0
    false_positive_count = 0.0
    for i in range(len(label_list)):
        if classify_res[i] == True:
            if label_list[i] == True:
                true_positive_count += 1
            else:
                false_positive_count += 1

    return true_positive_count/(true_positive_count + false_positive_count)

def cal_recall(label_list, classify_res):
    assert(len(label_list) == len(classify_res))
    true_positive_count = 0.0
    false_negative_count = 0.0
    for i in range(len(label_list)):
        if label_list[i] == True:
            if classify_res[i] == True:
                true_positive_count += 1
            else:
                false_negative_count += 1

    return true_positive_count/(true_positive_count + false_negative_count)

def write_to_logs(file_name, write_string):
    time_s = strftime("%Y-%m-%d %H:%M:%S ", gmtime())
    with open (file_name, "a+") as f:
        f.write(time_s + write_string + "\n")
