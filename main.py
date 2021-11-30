from preprocessing import Preprocessing
from data import Data

preprocess = Preprocessing
preprocess.doPreProcessing()

myData = Data
myData.loadPaths()
myData.classHistograms()
myData.splitData()

