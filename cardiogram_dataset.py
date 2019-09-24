from torch.utils.data import Dataset
from load_data import DataHelper
import torch
import random

class CardiogramDataset(Dataset):
    def __init__(self, list_files, labels):
        self.helper = DataHelper()
        self.list_files = list_files # list_files is the filename of training sample
        self.labels = labels

    def __getitem__(self, index):
        features = helper.get_features_from_txt(self.list_files[index])
        features = torch.FloatTensor(features)
        # print("index {} label {}".format(index, self.labels[index]))
        label = torch.FloatTensor([self.labels[index]])
        # print("float label {}".format(label))
        return features, label

    def __len__(self):
        return len(self.list_files)


def k_fold(k, list_files, labels):
    fold_num = int(len(list_files)/k)
    k_datas, k_labels = [], []

    for i in range(k-1):
        tmp_datas, tmp_labels = [], []
        for j in range(fold_num):

            r = random.randint(0, len(list_files)-1)
            tmp_datas.append(list_files[r])
            tmp_labels.append(labels[r])
            del list_files[r]
            del labels[r]
        k_datas.append(tmp_datas)
        k_labels.append(tmp_labels)

    k_datas.append(list_files)
    k_labels.append(labels)
    return k_datas, k_labels



helper = DataHelper()
files = helper.files
num_labels = helper.num_labels

datas, labels = k_fold(5, files, num_labels)
validation_data = datas[0]
validation_label = labels[0]

train_data = []
train_labels = []
for i in range(1, 5):
    train_data += datas[i]
    train_labels += labels[i]

dataset = CardiogramDataset(train_data, train_labels)
# print(len(dataset))
