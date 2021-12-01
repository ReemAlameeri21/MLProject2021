from preprocessing import Preprocessing
from data import Data
from train import Train
from analysis import Analysis
import matplotlib.pyplot as plt
from data import CheXpert

from torch.utils.data import DataLoader
import torch
from sklearn.utils import class_weight
import numpy as np
import torchvision.models as models
import torchvision
import torch.nn as nn
import torch.optim as optim


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


def calculate_cls_weight(trainData):
    labels = []

    for _, label in trainData:
        labels.append(label)

    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(labels), y=np.array(labels))
    class_weights = torch.tensor(class_weights, dtype=torch.float, device=device)

    print('class weights:', class_weights)
    return class_weights


seed = 1997
torch.cuda.manual_seed_all(seed)

d_net=models.densenet121(pretrained=False)
#d_net.classifier.in_features
num_ftrs_d=d_net.classifier.in_features
num_classes= 4
d_net.classifier= nn.Linear(num_ftrs_d, num_classes)
d_net= d_net.to(device)

loss_func_test= nn.CrossEntropyLoss()

loss_func_train= nn.CrossEntropyLoss(weight= calculate_cls_weight(myData.trainData))


optimizer = optim.AdamW(d_net.parameters(), lr=0.01)

epochs= 35
modelname='densenet121_baseline'
myTrain = Train
model,test_labels, pred_cls, pred_proba, train_loss, test_loss= myTrain.train_nn(d_net,
                                                                        train_loader,test_loader,loss_func_train,
                                                                        loss_func_test, optimizer,epochs, modelname, device)

myAnalysis = Analysis
myAnalysis.conf_mtrx (test_labels,pred_cls)
myAnalysis.evaluate_metrics (test_labels,pred_cls, myData.data)
myAnalysis.ROC_plot_AUC_score (test_labels,pred_proba,len(myData.data.classes), myData.data)


x_epoch = np.arange(35)
# print(test_loss)

plt.plot(x_epoch, train_loss, linestyle='--', label='Train loss')
plt.plot(x_epoch, test_loss, linestyle='--', label='Test loss')

plt.title('Learning curve- loss against epochs plot')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='best')

#instantiate resenet18 model object
resnet18_model= models.resnet18(pretrained=False)
# Fully connected layer input features:
input_ftrs= resnet18_model.fc.in_features
# number of output classes
num_classes= 4
#replace the fully connected layer to make it comaptible with our datset
resnet18_model.fc= nn.Linear(input_ftrs, num_classes)
# transfer the model to gpu
resnet18_model= resnet18_model.to (device)
# #instantiate crossentropy loss object

loss_func_test= nn.CrossEntropyLoss()

loss_func_train= nn.CrossEntropyLoss(weight= calculate_cls_weight(myData.trainData))

optimizer = optim.AdamW(d_net.parameters(), lr=0.01)

"""main"""
epochs= 35
modelname= 'resnet18_baseline'
model,test_labels, pred_cls, pred_proba, train_loss, test_loss=myTrain.train_nn(resnet18_model,
                                                                        train_loader,test_loader,
                                                                        loss_func_train,loss_func_test, optimizer,epochs, modelname, device)

myAnalysis.conf_mtrx (test_labels,pred_cls)
myAnalysis.evaluate_metrics (test_labels,pred_cls, myData.data)
myAnalysis.ROC_plot_AUC_score (test_labels,pred_proba,len(myData.data.classes), myData.data)

plt.plot(x_epoch, train_loss, linestyle='--', label='Train loss')
plt.plot(x_epoch, test_loss, linestyle='--', label='Test loss')

plt.title('Learning curve- loss against epochs plot')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='best')

resnet18_model= models.resnet18(pretrained=True)
input_ftrs= resnet18_model.fc.in_features
num_classes= 4
resnet18_model.fc= nn.Linear(input_ftrs, num_classes)
resnet18_model= resnet18_model.to (device)

loss_func_test= nn.CrossEntropyLoss()

loss_func_train= nn.CrossEntropyLoss(weight= calculate_cls_weight(myData.trainData))


optimizer = optim.AdamW(d_net.parameters(), lr=0.01)

modelname='resnet18_pretrainimagenet'
epochs= 25
model,test_labels, pred_cls, pred_proba, train_loss, test_loss=myTrain.train_nn(resnet18_model,
                                                                        train_loader,test_loader,
                                                                        loss_func_train,loss_func_test, optimizer,epochs, modelname, device)

myAnalysis.conf_mtrx (test_labels,pred_cls)
myAnalysis.evaluate_metrics (test_labels,pred_cls, myData.data)
myAnalysis.ROC_plot_AUC_score (test_labels,pred_proba,len(myData.data.classes), myData.data)

x_epoch = np.arange(25)

plt.plot(x_epoch, train_loss, linestyle='--', label='Train loss')
plt.plot(x_epoch, test_loss, linestyle='--', label='Test loss')

plt.title('Learning curve- loss against epochs plot')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='best')


d_net=models.densenet121(pretrained=True)
num_ftrs_d=d_net.classifier.in_features
num_classes= 4
d_net.classifier= nn.Linear(num_ftrs_d, num_classes)
d_net= d_net.to(device)

loss_func_test= nn.CrossEntropyLoss()

loss_func_train= nn.CrossEntropyLoss(weight= calculate_cls_weight(myData.trainData))


optimizer = optim.AdamW(d_net.parameters(), lr=0.01)

modelname='densenet121_pretrain_imagenet'
epochs= 25
model,test_labels, pred_cls, pred_proba, train_loss, test_loss=myTrain.train_nn(d_net,
                                                                        train_loader,test_loader,
                                                                        loss_func_train,loss_func_test, optimizer,epochs, modelname, device)

myAnalysis.conf_mtrx (test_labels,pred_cls)
myAnalysis.evaluate_metrics (test_labels,pred_cls, myData.data)
myAnalysis.ROC_plot_AUC_score (test_labels,pred_proba,len(myData.data.classes), myData.data)

x_epoch = np.arange(25)

plt.plot(x_epoch, train_loss, linestyle='--', label='Train loss')
plt.plot(x_epoch, test_loss, linestyle='--', label='Test loss')

plt.title('Learning curve- loss against epochs plot')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='best')


myCheXpert = CheXpert

batch_size = 32

train_loader= DataLoader(dataset= myCheXpert.trainDataset,
                         batch_size= batch_size,
                         shuffle =True,
                         num_workers = 8)
test_loader= DataLoader(dataset= myCheXpert.testDataset,
                        batch_size= batch_size,
                        shuffle =False,
                        num_workers = 8)


