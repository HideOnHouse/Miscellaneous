import os

import torch
import torch.nn.init
import torchvision
from torch.utils.data import DataLoader

from tqdm import tqdm
from matplotlib import pyplot as plt


class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 6, 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(6, 16, 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )

        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )

        self.layer4 = torch.nn.Sequential(
            torch.nn.Linear(128, 120),
            torch.nn.ReLU(),
            torch.nn.Linear(120, 2)
        )

    def forward(self, inp):
        out = self.layer1(inp)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.layer4(out)
        return out


def load_data():
    result = {}
    for target in os.listdir(r'result'):
        if target == 'mal' or target == 'benign':
            for file in os.listdir(os.path.join(r'C:\\', target)):
                path = os.path.join(r'C:\\', target, file)
                if target == 'mal':
                    result[path] = 1
                else:
                    result[path] = 0
    return result


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # hyper parameter
    leaning_rate = 0.001
    training_epoch = 50
    batch_size = 100
    training_ratio = 0.8

    trans = torchvision.transforms.Compose([torchvision.transforms.Grayscale(), torchvision.transforms.ToTensor()])
    train_data = torchvision.datasets.ImageFolder(root=r'image\\train', transform=trans)
    data_loader = torch.utils.data.DataLoader(train_data, batch_size=20, shuffle=True)
    model = CNN().to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=leaning_rate)
    total_batch = len(data_loader)

    print("Learning Start")
    for epoch in range(training_epoch):
        avg_cost = 0
        for data in tqdm(data_loader):
            imgs, labels = data
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            cost = criterion(out, labels)
            cost.backward()
            optimizer.step()

            avg_cost += cost / total_batch
        print('cost :', float(avg_cost))
    print("Leaning Finish")
    torch.save(model, r'C:\\result\model.pth')

    model = torch.load('result\\model.pth')
    model.eval()
    test_data = torchvision.datasets.ImageFolder(root=r'image\\test', transform=trans)
    data_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True)

    with torch.no_grad():
        correct = 0
        recall_point = 0
        mal_count = 0
        for data in tqdm(data_loader, bar_format='{l_bar}{bar:100}{r_bar}'):
            imgs, labels = data
            imgs = imgs.to(device)
            labels = labels.to(device)
            prediction = model(imgs)
            for i, j in zip(prediction, labels):
                p = 0
                if j == 1:
                    mal_count += 1
                if max(i) == i[0]:
                    p = 0
                else:
                    p = 1
                if p == j:
                    correct += 1
                    if p == 1:
                        recall_point += 1
        print("acc : ", correct / len(test_data) * 100, '%')
        print("recall : ", recall_point / mal_count * 100, '%')
