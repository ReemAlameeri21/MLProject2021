from preprocessing import Preprocessing
from data import Data
from train import Train
from analysis import Analysis
import matplotlib.pyplot as plt
from trainChex import TrainChex
from loadChex import LoadChex

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


#ImageNet, resnet

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


#ImageNet, densenet

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




#chexpert, resnet



res_chxtrans= models.resnet18(pretrained=False)
res_chxtrans.inplanes=64
res_chxtrans.conv1=nn.Conv2d(1,res_chxtrans.inplanes,kernel_size=7,stride=2,padding=3,bias=False)
input_ftrs= res_chxtrans.fc.in_features
num_classes= 14
res_chxtrans.fc= nn.Linear(input_ftrs, num_classes)

# transfer the model to gpu
res_chxtrans= res_chxtrans.to (device)

# #instantiate crossentropy loss object


loss_func_train= nn.CrossEntropyLoss()


optimizer = optim.SGD(res_chxtrans.parameters(), lr=0.01, momentum=0.9)


modelname='resnet18'
epochs= 100

myTrainCheX = TrainChex
myloadChex = LoadChex
pre_model= myTrainCheX.trainchxpert_nn(res_chxtrans, myloadChex.trainDataLoader,myloadChex.validDataLoader,loss_func_train, optimizer,epochs, modelname, device)

modelname='resnet18'
checkpoint= torch.load(f'{modelname}_model_pretrainedchxpert_checkpoint.pth.tar')

# Define the model:
res_chxtrans= models.resnet18()

# Change the model's structure:
# Change the model's features to make it compatible with grayscale images
res_chxtrans.inplanes=64
res_chxtrans.conv1=nn.Conv2d(1,res_chxtrans.inplanes,kernel_size=7,stride=2,padding=3,bias=False)
# Fully connected layer input features:
input_ftrs= res_chxtrans.fc.in_features
# number of output classes
num_classes= 14
#replace the fully connected layer to make it comaptible with our datset
res_chxtrans.fc= nn.Linear(input_ftrs, num_classes)


# Update the model's weights:
res_chxtrans.load_state_dict(checkpoint['model'])

# Save the model:
torch.save(res_chxtrans, (f'best_{modelname}_model_pretrainedchxpert.pth'))


pre_model= torch.load('/home/shahad.hardan/Downloads/ML701Prj/best_resnet18_model_pretrainedchxpert.pth')
input_ftrs= pre_model.fc.in_features
# number of output classes
num_classes= 4
#replace the fully connected layer to make it comaptible with our datset
pre_model.fc= nn.Linear(input_ftrs, num_classes)



#fine-tuning with our dataset

# transfer the model to gpu
pre_model= pre_model.to (device)

# instantiate crossentropy loss object
loss_func_test= nn.CrossEntropyLoss()

loss_func_train= nn.CrossEntropyLoss(weight= calculate_cls_weight(myData.trainData))


optimizer = optim.AdamW(d_net.parameters(), lr=0.01)

modelname='resnet18_downstream_chexpert'
epochs= 25
model,test_labels, pred_cls, pred_proba, train_loss, test_loss= myTrain.train_nn(pre_model,
                                                                        train_loader,test_loader,
                                                                        loss_func_train,loss_func_test, optimizer,epochs, modelname, device)

myAnalysis.conf_mtrx (test_labels,pred_cls)
myAnalysis.evaluate_metrics (test_labels,pred_cls, myData.data)
myAnalysis.ROC_plot_AUC_score (test_labels,pred_proba,len(myData.data.classes), myData.data)


#chexpert, densenet

dens_chxtrans= models.densenet121(pretrained=False)
num_init_features=64
dens_chxtrans.features.conv0=nn.Conv2d(1, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)

num_ftrs=dens_chxtrans.classifier.in_features
num_classes= 14
dens_chxtrans.classifier= nn.Linear(num_ftrs, num_classes)

dens_chxtrans= dens_chxtrans.to(device)

loss_func_train= nn.CrossEntropyLoss()

optimizer = optim.SGD(dens_chxtrans.parameters(), lr=0.01, momentum=0.9)

modelname='densenet121'

epochs= 100
pre_model=myTrainCheX.trainchxpert_nn(dens_chxtrans,myloadChex.trainDataLoader,myloadChex.validDataLoader,loss_func_train, optimizer,epochs, modelname, device)

modelname='densenet121'
checkpoint= torch.load(f'{modelname}_model_pretrainedchxpert_checkpoint.pth.tar')

dens_chxtrans= models.densenet121()

num_init_features=64
dens_chxtrans.features.conv0=nn.Conv2d(1, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)
num_ftrs=dens_chxtrans.classifier.in_features
num_classes= 14
dens_chxtrans.classifier= nn.Linear(num_ftrs, num_classes)



# Update the model's weights:
dens_chxtrans.load_state_dict(checkpoint['model'])

# Save the model:
torch.save(dens_chxtrans, (f'best_{modelname}_model_pretrainedchxpert.pth'))

model = torch.load('/home/shahad.hardan/Downloads/ML701Prj/best_densenet121_model_pretrainedchxpert.pth')
num_ftrs=model.classifier.in_features
num_classes= 4
# model.op_threshs = None # prevent pre-trained model calibration
model.classifier= nn.Linear(num_ftrs, num_classes)
model= model.to(device)


loss_func_train= nn.CrossEntropyLoss(weight= calculate_cls_weight(myData.trainData))
loss_func_test= nn.CrossEntropyLoss()

optimizer = optim.AdamW(d_net.parameters(), lr=0.01)

epochs= 25
modelname='densenet121-downstream_chexpert'
model,test_labels, pred_cls, pred_proba, train_loss, test_loss=myTrain.train_nn(model,
         train_loader,test_loader,
         loss_func_train, loss_func_test, optimizer,epochs, modelname, device)


myAnalysis.conf_mtrx (test_labels,pred_cls)
myAnalysis.evaluate_metrics (test_labels,pred_cls, myData.data)
myAnalysis.ROC_plot_AUC_score (test_labels,pred_proba,len(myData.data.classes), myData.data)

