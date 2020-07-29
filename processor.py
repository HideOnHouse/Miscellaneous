import os

import cv2
from tqdm import tqdm
import numpy as np

K = 1000


def make_bin_image_with_resize(root_path, file_extension):
    file_list = []
    excluded_list = []
    for path, dirs, files in os.walk(root_path):
        for file in files:
            if file.count(file_extension) != 0:
                file_list.append(os.path.join(path, file))
    for each in tqdm(file_list):
        with open(each, 'rb') as f:
            file_data = f.read()

        file_size = len(file_data)

        if file_size < 10 * K:
            image_width = 32
        elif file_size < 30 * K:
            image_width = 64
        elif file_size < 60 * K:
            image_width = 128
        elif file_size < 100 * K:
            image_width = 256
        elif file_size < 200 * K:
            image_width = 384
        elif file_size < 500 * K:
            image_width = 512
        elif file_size < 1000 * K:
            image_width = 768
        else:
            image_width = 1024

        image_height = file_size // image_width
        file_array = np.array(list(file_data[:image_width * image_height]))  # drop
        file_img = np.reshape(file_array, (image_height, image_width))
        file_img = np.uint8(file_img)
        try:
            file_img = cv2.resize(file_img, (32, 32), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(
                os.path.join('C:\\-\\-\\-\\-\\image\\' + foo,
                             each[each.rfind('\\') + 1:] + '.png'), file_img)
        except Exception as e:
            excluded_list.append(each)

    for exc in excluded_list:
        print(exc)


if __name__ == '__main__':
    root = ['mal\\-']
    for foo in root:
        if foo == 'mal\\-':
            make_bin_image_with_resize(foo, 'vir')
        else:
            make_bin_image_with_resize(foo, 'docx')
