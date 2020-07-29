import os
import csv
import shutil

from tqdm import tqdm


def main():
    with open(r'C:\-\-\-\-\-\train.csv', 'r') as csvfile:
        csv_data = csv.reader(csvfile)
        for row in tqdm(csv_data):
            file_path = 'C:\\-\\-\\-\\-\\-\\images\\' + row[0] + '.png'
            file_tag = row[1]
            try:
                if file_tag == 0:
                    shutil.move(file_path,
                                'C:\\-\\-\\-\\-\\-\\images\\train\\0\\' + row[
                                    0] + '.png')
                else:
                    shutil.move(file_path,
                                'C:\\-\\-\\-\\-\\-\\images\\train\\1\\' + row[
                                    0] + '.png')
            except Exception as e:
                pass


if __name__ == '__main__':
    main()
