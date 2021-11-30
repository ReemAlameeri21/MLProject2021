from PIL import Image
import numpy as np
from skimage import io
from skimage.io import imread
import torch
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt

class CXRDataSet(Dataset):

    def __init__(self, dataset, band, transform=None):
        self.dataset = self.checkchannels(dataset, band)
        self.transform = transform

    def checkchannels(self, dataset, band):

        if band == 'RGB':
            datasetRGB = []
            for index in range(len(dataset)):
                if Image.open(dataset[index][0]).getbands() == ('R', 'G', 'B'):
                    datasetRGB.append(dataset[index])
            return datasetRGB
        elif band == 'L':
            datasetG = []
            for index in range(len(dataset)):
                if Image.open(dataset[index][0]).getbands() == ('L',):
                    datasetG.append(dataset[index])
            return datasetG

    def croppingframe(self, index):  # Custom transform
        image = io.imread(self.dataset[index][0])
        nonzero_pixels = image[np.nonzero(image)]
        if nonzero_pixels.shape == (0,):
            return image
        min = np.array(np.nonzero(image)).min(axis=1)
        max = np.array(np.nonzero(image)).max(axis=1)
        return image[min[0]:max[0] + 1, min[1]:max[1] + 1]

    def __getitem__(self, index):  # allows us to index our instance

        image = self.croppingframe(index)
        image = Image.fromarray(image)
        y_label = torch.tensor(self.dataset[index][1])
        if self.transform:
            return self.transform(image), y_label

        return image, y_label

    def __len__(self):
        return len(self.dataset)

class Data:

    def __init__(self, data):
        path_to_rgbdata = 'COVID_RGB'
        data = ImageFolder(path_to_rgbdata)
        self.data = data

    def loadPaths(self):
        data = self.data

        #  data variable attributes?
        #print(data)
        print('Total number of datapoints:', len(data.imgs))
        # shows the number of datapoints and root location
        print('---------------------------')
        #print(data.imgs[0:10])
        # list of tuples, each of which contains the image path and its label

        # print(data.targets[0:10])
        # only labels [0,1,2,...]

        print("Dataset classes:", data.classes)
        # the name of the classes [Normal , covid, ..]

    def classHistograms(self):
        data = self.data
        # Show imbalance in dataset
        counter = [data.targets.count(l) for l in set(data.targets)]
        labels = data.classes
        print(counter, labels)  # print perecentages
        plt.bar(labels, counter, width=0.2)
        plt.xlabel("Number of images per class")
        plt.ylabel("No. of images ")
        plt.title("Class")
        plt.show()

    def splitData(self):
        data = self.data
        # Split dataset into train test sets:
        trainData, testData, trainLabel, testLabel = train_test_split(data.imgs, data.targets, test_size=0.1, random_state=0, stratify=data.targets)

