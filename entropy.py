import os
import csv
import math
import collections

from tqdm import tqdm


def entropy(data):
    e = 0
    counter = collections.Counter(data)
    l = len(data)
    for count in counter.values():
        p_x = count / l
        e += - p_x * math.log2(p_x)
    return e


def get_entropy(folder_path):
    e_dict = {}
    with open('entropy_result.txt', 'r+') as res:
        res.write(folder_path + '\n')
        for path, dirs, files in os.walk(folder_path):
            for file in files:
                if sum(file.count(i) for i in ['jpg', 'png', 'jpeg', 'gif']) == 0:
                    with open(path + os.sep + file, 'rb') as f:
                        e_dict[file] = entropy(f.read())
        for item in sorted(e_dict.items(), key=lambda x: x[1], reverse=True):
            if item[1] >= 5.8:
                res.write(str(item) + '\n')
        if len(e_dict) != 0:
            res.write('=' * 40 + '\n')


def main():
    root = r'C:\-\-\-\-\-\benign'
    # ================ Read CSV =====================
    # with open(r'C:\-\-\-\-\md5_list.csv', 'r') as f:
    #     csv_data = csv.reader(f)
    #     for row in csv_data:
    #         print(root + os.sep + row[0])
    #         get_entropy(root + os.sep + row[0])
    for sub_folder in tqdm(os.listdir(root)):
        get_entropy(root + os.sep + sub_folder)


if __name__ == '__main__':
    main()
