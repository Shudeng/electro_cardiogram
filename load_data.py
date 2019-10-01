import numpy as np
import torch
import Constant
import os
# from cardiogram_dataset import CardiogramDataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import random

class CardiogramDataset(Dataset):
    def __init__(self, list_files, labels):
        self.helper = DataHelper()
        self.list_files = list_files # list_files is the filename of training sample
        self.labels = labels

    def __getitem__(self, index):
        features = self.helper.get_features_from_txt(self.list_files[index])
        features = torch.FloatTensor(features)
        # print("index {} label {}".format(index, self.labels[index]))
        label = torch.FloatTensor([self.labels[index]])
        # print("float label {}".format(label))
        return features, label

    def __len__(self):
        return len(self.list_files)

class DataHelper:
    def __init__(self):
        """
        the key of the patient_label_hash is filename of each patient, in "id.txt" form.
        the value of the patient_label_hash is the patient features including age, sex and arrythmia
        """
        self.patient_label_hash = self.create_patient_label_hash()

        #match label to its corresponding num
        self.label_to_num_hash = self.label_to_num_hash()
        # match num to its corresponding label.
        self.num_to_label_hash = self.num_to_label_hash()
        self.files = self.get_files()
        self.num_labels = self.get_num_labels()
        self.labels = self.get_labels()


    def get_features_from_txt(self, txt_file):
        with open(Constant.TRAIN_DATA_PATH + txt_file) as f:
            lines = f.readlines()
        lines = lines[1:]
        features = []
        for line in lines:
            line = line.rstrip()
            feature = line.split(" ")
            features.append(feature)
        features = [[float(str) for str in item1] for item1 in features]
        return features

    def create_patient_label_hash(self):
        with open(Constant.PATIENT_LABEL_FILE) as f:
             lines = f.readlines()
        patient_label_hash={}
        for line in lines:
            line = line.rstrip()
            line_items = line.split("\t")
            labels = []
            for item in line_items[1:]:
                if not item=="":
                    labels.append(item)
            patient_label_hash[line_items[0]]=labels
            # print(line_items[0], labels)
        return patient_label_hash

    def get_patient_feature_and_label(self, txt_name):
        """
        Here we ignore age and sex feature.
        However these two features may have influence on prediction.
        :param txt_name:
        :return: combine features and label/arrythmia
        """
        features = self._get_features_from_txt(txt_name)
        labels = self.patient_label_hash[txt_name]
        num_labels = []
        for label in labels:
            if label in self.label_to_num_hash:
                num_labels.append(self.label_to_num_hash[label])
        return features, num_labels, labels

    def label_to_num_hash(self):
        with open(Constant.ARRYTHMIA_FILE) as f:
            lines = f.readlines()
        label_to_num_hash = {}
        for line, i in zip(lines, range(len(lines))):
            line = line.rstrip()
            label_to_num_hash[line] = i
        return label_to_num_hash

    def num_to_label_hash(self):
        with open(Constant.ARRYTHMIA_FILE) as f:
            lines = f.readlines()
        num_to_label_hash = {}
        for line, i in zip(lines, range(len(lines))):
            line = line.rstrip()
            num_to_label_hash[i] = line
        return num_to_label_hash
    
    def get_label_by_txtname(self, txt_name):
        return self.patient_label_hash[txt_name]

    def get_num_label_by_txtname(self, txt_name):
        labels = self.get_label_by_txtname(txt_name)
        num_labels = []
        for label in labels:
            if label.isdigit() or label=='MALE' or label=='FEMALE':
                continue
            else:
                num_labels.append(self.label_to_num_hash[label])

        return num_labels

    def prepare_filenames_and_labels(self):
        files = [f for f in os.listdir(Constant.TRAIN_DATA_PATH) if os.path.isfile(os.path.join(Constant.TRAIN_DATA_PATH, f))]
        labels = [self.get_label_by_txtname(f) for f in files]

        return files, labels

    def get_files(self):
        """
        :return:
        """
        train_path = Constant.TRAIN_DATA_PATH
        files = [f for f in os.listdir(train_path) if os.path.isfile(os.path.join(train_path, f))]
        return files

    def get_num_labels(self):
        num_labels = [self.get_num_label_by_txtname(file) for file in self.files]
        return num_labels

    def get_num_label_i(self, i):
        """
        Get the ith indicator label
        1 if the patient has ith arrythmia
        0 otherwise
        :param i:
        :return:
        """
        num_labels = self.get_num_labels()
        indicator_labels = []
        for label in num_labels:
            if i in label:
                indicator_labels.append(1)
            else:
                indicator_labels.append(0)
        return indicator_labels

    def get_labels(self):
        labels = (self.get_label_by_txtname(file) for file in self.files)

    def k_fold(aelf, k, list_files, labels):
        fold_num = int(len(list_files) / k)
        k_datas, k_labels = [], []

        for i in range(k - 1):
            tmp_datas, tmp_labels = [], []
            for j in range(fold_num):
                r = random.randint(0, len(list_files) - 1)
                tmp_datas.append(list_files[r])
                tmp_labels.append(labels[r])
                del list_files[r]
                del labels[r]
            k_datas.append(tmp_datas)
            k_labels.append(tmp_labels)

        k_datas.append(list_files)
        k_labels.append(labels)
        return k_datas, k_labels

    def get_train_and_validation_iter(self):
        files = self.files
        num_labels = self.get_num_label_i(0)  ## indicator patient has the 0th arrythmia or not
        # for file, num_label_0 in zip(files, num_labels):
        #     print(file+"\t"+str(num_label_0))

        datas, labels = self.k_fold(5, files, num_labels)
        validation_data = datas[0]
        validation_label = labels[0]

        train_data = []
        train_labels = []
        for i in range(1, 5):
            train_data += datas[i]
            train_labels += labels[i]

        params = {
            'batch_size': 64,
            'shuffle': True,
            'num_workers': 6
        }

        dataset = CardiogramDataset(train_data, train_labels)

        train_iter = DataLoader(dataset, **params)
        validation_iter = DataLoader(CardiogramDataset(validation_data, validation_label), **params)
        return train_iter, validation_iter

