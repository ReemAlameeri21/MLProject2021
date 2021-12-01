from PIL import Image
import numpy as np
from skimage import io
from skimage.io import imread
import torch
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

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

    data = None
    mean = None
    std = None
    trainDataset = None
    testDataset = None
    trainData = None
    testData = None

    def __init__(self, data, mean, std, trainDataset, testDataset):
        self.data = data
        self.mean = mean
        self.std = std
        self.trainDataset = trainDataset
        self.testDataset = testDataset

    def loadPaths(self):
        path_to_rgbdata = 'COVID_RGB'
        self.data = ImageFolder(path_to_rgbdata)
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

    def handleData(self):
        data = self.data
        # Split dataset into train test sets:
        trainData, testData, trainLabel, testLabel = train_test_split(data.imgs, data.targets, test_size=0.1, random_state=0, stratify=data.targets)
        self.trainData = trainData
        self.testData = testData
        # Evaluate mean and std of images:
        # mean and std are dependent on image size.
        Data_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                              transforms.ToTensor()])

        CXR_dataset = CXRDataSet(trainData, 'RGB',
                                 transform=Data_transforms)

        CXR_loader = torch.utils.data.DataLoader(CXR_dataset,
                                                 batch_size=32, num_workers=4, shuffle=False)

        mean, std = self.get_mean_and_std(CXR_loader)

        self.mean = mean
        self.std = std

        Data_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
        ])
        # Important note: if we are going to use transforms.Resize
        # The image must be read using PIL.Image.open, otherwise it won't work

        train_set = CXRDataSet(trainData, 'RGB',
                               transform=Data_transforms)
        self.trainDataset = train_set

        test_set = CXRDataSet(testData, 'RGB',
                              transform=Data_transforms)

        self.testDataset = test_set

        print("Size of train dataset: ", len(train_set))
        print("Size of test dataset: ", len(test_set))

        # train_set, test_set = torch.utils.data.random_split(CXR_dataset, [12123,3030])
        # Our dataset contains 15153 images

    def get_mean_and_std(loader):
        mean = 0.
        std = 0.
        Total_images_count = 0.
        for images, _ in loader:
            Total_images_count += images.size(0)
            # or you can use images.shape[0]
            # print(images.shape) B xC X H X W
            images = images.view(images.size(0), images.size(1), -1)
            # print(images.shape) B x C x (H*W)
            mean += images.mean(2).sum(0)
            # find the mean of each channel
            # in all images of the batch, axis=2
            # sum the means of corresponding
            # channels of the batch images, axis=0
            std += images.std(2).sum(0)

        mean /= Total_images_count
        std /= Total_images_count

        return mean, std

    def show_transformed_images(self):
        dataset = self.trainDataset
        data = self.data

        loader = torch.utils.data.DataLoader(dataset, batch_size=6, shuffle=True)
        batch = next(iter(loader))
        images, labels = batch
        # print(images.shape) # BxCxWxH
        images = images.numpy().transpose((0, 2, 3, 1))
        # convert the images from torch to numpy and transpose
        # print(images.shape)# BxWxHxC
        mean_arr = np.array(self.mean)
        std_arr = np.array(self.std)
        images = std_arr * images + mean_arr
        # When we normalize our data, it gets shifted, such that
        # its mean becomes 0 and std. dev. of 1 i.e. the data will have negative portion.
        images = np.clip(images, 0, 1)
        fig, axes = plt.subplots(2, 3, figsize=(7, 5),
                                 subplot_kw={'xticks': [], 'yticks': []},
                                 gridspec_kw=dict(hspace=0.1, wspace=0.05))
        for i, ax in enumerate(axes.flat):
            ax.imshow(images[i], cmap='gray')
            ax.set_xlabel('label: {}'.format(data.classes[labels[i]]))


class CheXpert: