from preprocessing import Preprocessing
from data import Data
from torch.utils.data import DataLoader
import torch

preprocess = Preprocessing
preprocess.doPreProcessing()

myData = Data
myData.loadPaths()
myData.classHistograms()
myData.handleData()


batch_size = 32

train_loader= DataLoader(dataset= myData.trainDataset,
                         batch_size= batch_size,
                         shuffle =True,
                         num_workers = 8)
test_loader= DataLoader(dataset= myData.testDataset,
                        batch_size= batch_size,
                        shuffle =False,
                        num_workers = 8)

#Show transformed images
myData.show_transformed_images()

#Set Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.is_available()

