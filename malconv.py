import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm


class MalConv(nn.Module):
    def __init__(self, window_size=500):
        super(MalConv, self).__init__()

        self.embed = nn.Embedding(32, 8, padding_idx=0)

        self.conv_1 = nn.Conv1d(4, 32, window_size, stride=window_size, bias=True)
        self.conv_2 = nn.Conv1d(4, 32, window_size, stride=window_size, bias=True)

        self.pooling = nn.MaxPool2d(2)

        self.fc_1 = nn.Linear(48, 128)
        self.fc_2 = nn.Linear(128, 2)

        self.sigmoid = nn.Sigmoid()
        # self.softmax = nn.Softmax()

    def forward(self, x):
        # Channel first
        x = torch.transpose(x, -1, -2)

        cnn_value = self.conv_1(x.narrow(-2, 0, 4))
        gating_weight = self.sigmoid(self.conv_2(x.narrow(-2, 4, 4)))

        x = cnn_value * gating_weight
        x = self.pooling(x)

        x = x.view(-1, 48)
        x = self.fc_1(x)
        x = self.fc_2(x)
        # x = self.sigmoid(x)

        return x


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # hyper parameter
    leaning_rate = 0.001
    training_epoch = 50
    batch_size = 100
    training_ratio = 0.8

    trans = torchvision.transforms.Compose([torchvision.transforms.Grayscale(), torchvision.transforms.ToTensor()])
    train_data = torchvision.datasets.ImageFolder(root=r'image\\train', transform=trans)
    data_loader = torch.utils.data.DataLoader(train_data, batch_size=4, shuffle=True)
    model = MalConv(window_size=5).to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=leaning_rate)
    total_batch = len(data_loader)

    print("Learning Start")
    for epoch in range(training_epoch):
        avg_cost = 0
        for data in tqdm(data_loader):
            imgs, labels = data
            imgs = imgs.view(4, 32, -1).to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            cost = criterion(out, labels)
            cost.backward()
            optimizer.step()

            avg_cost += cost / total_batch
        print('cost :', float(avg_cost))
    print("Leaning Finish")
    torch.save(model, r'C:\-\-\-\-\result\model2.pth')

    model = torch.load('result\\model2.pth')
    model.eval()
    test_data = torchvision.datasets.ImageFolder(root=r'image\\test', transform=trans)
    data_loader = torch.utils.data.DataLoader(test_data, batch_size=5, shuffle=True)

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
