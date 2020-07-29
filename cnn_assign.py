import os
import csv

from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.init
import torchvision

import torch.nn.functional as F

from torchvision.transforms import functional
from torch.utils.data import DataLoader


class CustomDataset(torch.utils.data.Dataset):
    # pretreatment of data
    def __init__(self, test=False):
        print("Loading data...")
        self.x_data = []
        self.y_data = []
        if test:
            postfix = 'test.csv'
        else:
            postfix = 'train.csv'
        with open(r'C:\-\-\-\-\-' + os.sep + postfix, 'r') as csv_file:
            csv_data = csv.reader(csv_file)
            temp = []
            for row in csv_data:
                temp.append(row[0])
                self.y_data.append([int(row[1])])
        for file in tqdm(temp[:1000]):
            image_path = r'C:\-\-\-\-\-\-' + os.sep + file + '.png'
            try:
                with open(image_path) as f:
                    img = Image.open(image_path)
                    img = img.convert("RGB")
                    t = torchvision.transforms.functional.to_tensor(img)
                    self.x_data.append(t)
            except FileNotFoundError as e:
                print(e)
        self.y_data = torch.FloatTensor(self.y_data[:1000])

    # when len() is called
    def __len__(self):
        return len(self.x_data)

    # when dataset[i] called
    def __getitem__(self, item):
        x = self.x_data[item]
        y = self.y_data[item]
        return x, y


class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
            torch.nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3, stride=2, padding=0, dilation=1, ceil_mode=False),
            torch.nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features=9216, out_features=4096, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=4096, out_features=4096, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=4096, out_features=1, bias=True),
        )

    def forward(self, x):
        out = self.layer1(x)
        out = out.view(-1, 6 * 6 * 256)
        out = self.classifier(out)
        return torch.sigmoid(out)


def bin_to_img(file_path, img_size=224):
    """

    :param file_path: target file to convert img
    :param img_size: desired size of img
    :return: img array that made from binary file
    """

    with open(file_path, 'rb') as f:
        file_data = f.read()

    file_size = len(file_data)

    if file_size < 10000:
        image_width = 32
    elif file_size < 30000:
        image_width = 64
    elif file_size < 60000:
        image_width = 128
    elif file_size < 100000:
        image_width = 256
    elif file_size < 200000:
        image_width = 384
    elif file_size < 500000:
        image_width = 512
    elif file_size < 1000000:
        image_width = 768
    else:
        image_width = 1024

    image_height = file_size // image_width
    file_array = np.array(list(file_data[:image_width * image_height]))  # drop rest
    file_img = np.reshape(file_array, (image_height, image_width))
    try:
        file_img = cv2.resize(file_img, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
        return file_img
    except Exception as e:
        print(e)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # hyper parameter
    leaning_rate = 0.001
    training_epoch = 50
    batch_size = 5

    model = CNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=leaning_rate)
    total_batch = -1  # TBD

    train_set = CustomDataset()
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)

    # Train
    model.train()
    for epoch in range(training_epoch):
        avg_cost = 0
        for data in tqdm(train_loader):
            imgs, labels = data
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            cost = F.binary_cross_entropy(out, labels).to(device)
            cost.backward()
            optimizer.step()

            avg_cost += cost / total_batch
        print('cost :', float(avg_cost))
    torch.save(model, r'C:\-\model.pth')

    # Test
    test_set = CustomDataset(test=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=len(test_set), shuffle=True)
    model = torch.load(r'C:\-\model.pth')
    model.eval()

    with torch.no_grad():
        for data in tqdm(test_loader, bar_format='{l_bar}{bar:100}{r_bar}'):
            imgs, labels = data
            imgs = imgs.to(device)
            labels = labels.to(device)
            prediction = model(imgs)


if __name__ == '__main__':
    main()
