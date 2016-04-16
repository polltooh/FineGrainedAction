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
        for i in range(len(data_list)):
            d_l = data_list[i].split(" ")
            if (int(float(d_l[1])) < 5):
                print(d_l[1])
                image = cv2.imread(d_l[0])
                cv2.imwrite("temp/%08d.jpg"%(count), image)
                count = count + 1
                cv2.imshow("res", image)
                cv2.waitKey(100)


if __name__ == "__main__":
    read_file("test_res.txt")
