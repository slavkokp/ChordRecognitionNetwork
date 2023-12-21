import json
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import random

# remove dataset entries that include only 14 to 19 classes on all strings and if include both 14+ and less than 14, then set 14+ to 0
def filter_classes(section):
    code = torch.FloatTensor(section["strings"]).argmax(dim=1)
    keep = True
    for i in range(6):
        if code[i] >= 14:
            keep = False
        if code[i] < 14 and code[i] > 0:
            keep = True
            break
    return keep

# remove dataset entries that include 14 to 19 classes on any string
def filter_classes_hard(section):
    code = torch.FloatTensor(section["strings"]).argmax(dim=1)
    keep = True
    for i in range(6):
        if code[i] >= 14:
            keep = False
            break
    return keep

# random drop half
def keep(label):
    if label == 17:
        n = random.randint(0, 1)
        if n < 1:
            return False
    return True

class CustomImageDataset(Dataset):
    # labels should be stored as a dict by labels_path, you can do that with torch.save(new_labels, "new_labels.pth")
    def __init__(self, annotations_file=None, transform=None, target_transform=None, drop_half_empty=False, labels=False, labels_path="labels.pth"):
        if annotations_file != None:
            with open(annotations_file, 'r') as file:
                self.data = json.load(file)
            self.data["sections"] = list(filter(lambda section: len(section["spectrogram"]) == 864, self.data["sections"]))
            
            # self.labels = [None] * len(self.data["sections"])
            if labels:
                self.labels = torch.load(labels_path)

            if drop_half_empty:
                random.seed(42)
                keep_arr = [keep(l) for l in self.labels]
                new_sections = []
                new_labels = []
                for i in range(len(self.data["sections"])):
                    if keep_arr[i]:
                        new_sections.append(self.data["sections"][i])
                        new_labels.append(self.labels[i])
                self.labels = new_labels
                self.data["sections"] = new_sections

            
            # filter dataset to get rid of classes 14+
            #self.data["sections"] = list(filter(filter_classes, self.data["sections"]))

            for m in range(len(self.data["sections"])):

                # shorten vectors from len 19 to 14 if model for 14 classes
                #self.data["sections"][m]["strings"] = [strg[:14] for strg in self.data["sections"][m]["strings"]]
                self.data["sections"][m]["spectrogram"] = np.repeat(np.array(self.data["sections"][m]["spectrogram"]).reshape(1, 96, 9), 3, 0)

                
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data["sections"])

    def __getitem__(self, idx):
        spectrogram = torch.div(torch.FloatTensor(self.data["sections"][idx]["spectrogram"]), 255)
        labels = torch.FloatTensor(self.data["sections"][idx]["strings"])
        return spectrogram, labels

    def get_labels(self):
        return self.labels

    def get_specific_labels(self, indexes):
        return [self.labels[idx] for idx in indexes]