import os
import numpy as np
import matplotlib.pyplot as plt
import torchvision.datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.autograd import Variable
from matplotlib.image import imread
from sklearn import preprocessing
from torch import float32
from torch import int64
#This file transfers 28×28 pixel images to a data tensor and a target tensor
COOKED_DIR = 'E:/photos/' # the file where you saved your the images
COOKED_DIR2 = 'E:/tensor/' # the file where you save the tensors

#Generate a tensor with a length of 1500, initializing each element to 0
Y = torch.zeros(1500,dtype=int64)
#Divide the tensor into 10 parts
slices = torch.split(Y, 150)
#Assign values to each portion separately
for i, s in enumerate(slices):
    s.fill_(2 * i + 1)
torch.save(Y, COOKED_DIR2+"myTensorY.pt")
print("save myTensorY.pt")
#check the tensor
D=torch.load(COOKED_DIR+"myTensorY.pt")
print(type(D))
print(D[150])
print(D.shape)
print(D.dtype)

def audiodatabuild():
    X_mother_list=[]
    for i in range(0,1500):
        name = str(i)
        img = (1-imread(COOKED_DIR+"photo28"+name+".jpg"))[::-1, :, 0].T
        min_max_scaler = preprocessing.MinMaxScaler()
        x_minmax = min_max_scaler.fit_transform(img)
        x_minmax_list=x_minmax.reshape(-1)
        X_mother_list.append(x_minmax_list)
        if(i%100==0):
            print(i)
    X_mother=np.array(X_mother_list)       
    X1 = X_mother.reshape(1500,1,28,28)
    X2 = torch.from_numpy(X1)
    X = X2.type(torch.float32)
    torch.save(X, COOKED_DIR2+"myTensorX.pt")
    print("save myTensorX.pt")



audiodatabuild()
C=torch.load(COOKED_DIR2+"myTensorX.pt")
#check the tensor
print(type(C))
print(C[1][0][15][16])
print(C.shape)
print(C.dtype)
