from preprocessing import Preprocessing
from data import Data
from torch.utils.data import DataLoader

preprocess = Preprocessing
preprocess.doPreProcessing()

myData = Data
myData.loadPaths()
myData.classHistograms()
myData.splitData()
myData.handleData()


batch_size = 32

train_loader= DataLoader(dataset= train_set,
                         batch_size= batch_size,
                         shuffle =True,
                         num_workers = 8)
test_loader= DataLoader(dataset= test_set,
                        batch_size= batch_size,
                        shuffle =False,
                        num_workers = 8)

myData.show_transformed_images()

