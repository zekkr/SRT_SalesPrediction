# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022/7/17 10:27
@file: Sales_forecast.py
@author: Zhuerk
"""

import pandas as pd
import numpy as np
#=== to these packages
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from sklearn.model_selection import train_test_split


import torch
from torch.utils.data import DataLoader, Dataset
print(torch.cuda.is_available())
print(torch.__version__)
#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = '1'


data_set = np.loadtxt(open("train.csv", 'r'), delimiter=',', skiprows=1, usecols=[3], dtype=int).reshape(1826, -1)
train_dates = np.loadtxt(open("train.csv", 'r'), delimiter=',', skiprows=1, usecols=[0], dtype=str)[0:1826,]
train_dates = pd.to_datetime(train_dates)
#print(train_dates)
#print(data_set[0]) # First day's Sales for 500 goods
# print(data_set[:,0]) # Sales series for good of id=(1,1) 
# print(data_set.shape)

#data_for_plot = pd.DataFrame(data_set[:,1:5])
#data_for_plot.plot.line()
#plt.savefig("Line.png")
#plt.clf()

# LSTM uses sigmoid and tanh that are sensitive to magnitude so values need to be normalized
# normalize the dataset
scaler = StandardScaler()
scaler = scaler.fit(data_set)
data_set_scaled = scaler.transform(data_set)
# print(data_set_scaled[0])
 
trainX = []
trainY = []

n_future = 1 # Number of days we want to predict into the future
n_past = 30 # Number of past days we want to use to predict the future

# forecast the first series
#=== three lines below
for i in range(n_past, len(data_set_scaled) - n_future + 1):
    trainX.append(data_set_scaled[i - n_past:i, 0:data_set_scaled.shape[1]])
    trainY.append(data_set_scaled[i + n_future -1:i + n_future, 0:data_set_scaled.shape[1]])

trainX = np.array(trainX)
trainY = np.array(trainY)

x_train, x_valid, y_train, y_valid = train_test_split(trainX, trainY, test_size=0.2)

print('x_train shape == {}.'.format(x_train.shape))
print('x_valid shape == {}.'.format(x_valid.shape))
print('y_train shape == {}.'.format(y_train.shape))
print('y_valid shape == {}.'.format(y_valid.shape))


batch_size = 32

class myDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.x_train = np.array(x_train, dtype=np.float32)
        self.y_train = np.array(y_train, dtype=np.float32)

    def __getitem__(self,index):
        return (self.x_train[index],self.y_train[index])

    def __len__(self):
        return self.x_train.shape[0]
    
class myValidDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.x_valid = np.array(x_valid, dtype=np.float32)
        self.y_valid = np.array(y_valid, dtype=np.float32)

    def __getitem__(self,index):
        return (self.x_valid[index],self.y_valid[index])

    def __len__(self):
        return self.x_valid.shape[0]

class myLSTM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.LSTM=torch.nn.LSTM(500, 256, 3, bias=True, batch_first=True)
        self.LSTMdropout=torch.nn.Dropout(p=0.2)
        self.fc = torch.nn.Linear(n_past*256, 500)
        #=== self.fc=torch.nn.Linear(256*batch_size,500,bias=True) # previous
    
    def forward(self,x): 
        outputs, hidden = self.LSTM(x)
        # hidden = hidden[0]  #=== comment this line
        # outputs = self.LSTMdropout(outputs) #=== remove dropout layer
        outputs = outputs.reshape(outputs.shape[0], -1)
        outputs = self.LSTMdropout(outputs)
        final = self.fc(outputs)
        return final

train_dataset = myDataset()
valid_dataset = myValidDataset() ###
#=== AttributeError: 'myDataset' object has no attribute 'shape'
# print('train_datset shape == {}.'.format(train_dataset.shape))
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
model = myLSTM()
device=torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
model = model.to(device)
criterion=torch.nn.MSELoss()
optimizer=torch.optim.Adam(model.parameters(),lr=1e-4) 


def train(epoch):
    model.train()
    #=== 打印每个epoch的平均loss, 而不只是最后一个batch的loss
    loss_list = []  #===
    for batch_idx, data in enumerate(train_dataloader):
        inputs, labels = data
        inputs ,labels = inputs.to(device), labels.to(device)
        outputs=model(inputs)
        labels = labels.reshape(outputs.shape)
        loss=criterion(outputs,labels)
        loss_list.append(loss.item())  #===
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    ep_loss = (sum(loss_list)/len(loss_list)) if len(loss_list) else -1
    print("Epoch",epoch,"MSE loss:",ep_loss)
    return ep_loss

def validate():
    model.eval()
    with torch.no_grad():
        loss_list = []
        for batch_idx, data in enumerate(valid_dataloader):
            inputs, labels = data
            inputs ,labels=inputs.to(device), labels.to(device)
            outputs=model(inputs)
            labels = labels.reshape(outputs.shape)
            loss=criterion(outputs,labels)
            loss_list.append(loss.item())  #===
        
        ep_loss = (sum(loss_list)/len(loss_list)) if len(loss_list) else -1
        print("Validation MSE loss:", ep_loss)
    
    return ep_loss
        
        

train_loss_list = []
val_loss_list = []
max_epoches = 300
val_every_epoches = 10

for epoch in range(max_epoches): 
    tr_loss = train(epoch)
    train_loss_list.append(tr_loss)
    # MSE_list.append()
    if epoch % val_every_epoches == val_every_epoches-1:
        val_loss = validate()
        val_loss_list.append(val_loss)
        
## Loss Plot
plt.figure(figsize=(8,8))
plt.plot(range(max_epoches), train_loss_list)
plt.plot(np.arange(val_every_epoches-1, max_epoches, val_every_epoches), val_loss_list)
plt.legend(['train', 'val'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.savefig('/home2/zhuerkang/Document/data_forecast/loss.png')
plt.close()

## Forecasting
n_future = 90 # Predict for three month
forecast_period_dates = pd.date_range(list(train_dates)[-1], periods=n_future, freq='1d').tolist()

model.eval()
forecast_list = []
start_x = torch.tensor(trainX[-1],dtype=torch.float).to(device)
start_x = start_x.unsqueeze(0)
print(start_x.shape)

for i in range(n_future):
    forecast = model(start_x)
    forecast_list.append(forecast)
    start_x = torch.concat([start_x[0,1:,:],forecast],dim=0)
    start_x = start_x.unsqueeze(0)

forecast_data = torch.concat(forecast_list,dim=0).cpu().detach().numpy()
print(forecast_data.shape)
#forecast_copies = np.repeat(forecast_data, data_set.shape[1], axis=-1)
y_pred_future = scaler.inverse_transform(forecast_data)
print(y_pred_future.shape)

#print(y_pred_future[0])
df_pred_future = y_pred_future.reshape(90*500,1,order = 'F')
df_future = pd.DataFrame(df_pred_future)
writer = pd.ExcelWriter('pred.xlsx')
df_future.to_excel(writer,'page_1',float_format='%d',index=False)  
writer.save()  

# Convert timestamp to date
forecast_dates = []
for time_i in forecast_period_dates:
    forecast_dates.append(time_i.date())
    
dat_forecast = pd.DataFrame({'Date':np.array(forecast_dates), 'Sales11':y_pred_future[:,0]})
dat_forecast["Date"]=pd.to_datetime(dat_forecast["Date"])

original = pd.DataFrame({'Date':train_dates, 'Sales11':data_set[:,0]})
original = original.loc[original['Date'] >= '2015-12-31']

sns.lineplot(original['Date'], original['Sales11'])
sns.lineplot(dat_forecast['Date'], dat_forecast['Sales11'])
sns.set_style('whitegrid')
plt.legend(labels=["original","forecast"])
plt.savefig("forecast.png")
plt.clf()

####  对比过去365天气的实际值与预测值
x_orig = torch.tensor(trainX[-365:],dtype=torch.float).to(device)
y_orig = trainY[-365:].reshape(-1,500)
y_pred = model(x_orig).cpu().detach().numpy()
print(y_orig.shape)
print(y_pred.shape)
tmp = np.linspace(0,364,365)

n_fit = 365 # Predict for three month
fit_period_dates = pd.date_range(list(train_dates)[-1], periods=n_fit, freq='1d').tolist()
fit_dates = []
for time_i in fit_period_dates:
    fit_dates.append(time_i.date())
    
dat_fit = pd.DataFrame({'Date':np.array(fit_dates), 'Sales11':y_pred[:,0]})
dat_fit["Date"]=pd.to_datetime(dat_fit["Date"])

dat_orig = pd.DataFrame({'Date':np.array(fit_dates), 'Sales11':y_orig[:,0]})
dat_orig["Date"]=pd.to_datetime(dat_orig["Date"])

sns.lineplot(dat_orig['Date'], dat_orig['Sales11'])
sns.lineplot(dat_fit['Date'], dat_fit['Sales11'])
sns.set_style('whitegrid')
plt.legend(labels=["original","fitted"])
plt.savefig("fit.png")
plt.clf()

exit()

# end