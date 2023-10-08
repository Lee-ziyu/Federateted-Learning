# data processing module
import numpy as np
import pandas as pd
from utils_sub import *

# federated learning module
from models import *
from FedAvg import *

# others
import os
import torch


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load data
train_set, test_set = dataloader_adult()
#传输过来的data已经被数字化
#归一化
num_columns=['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status',
              'occupation', 'relationship','race', 'sex', 'capital_gain', 'capital_loss',
              'hours_per_week', 'native_country']
num_mean=train_set[num_columns].mean()
num_std=train_set[num_columns].std()
num_normal=(train_set[num_columns]-num_mean)/num_std
df_x=train_set.drop(columns=num_columns)
df_x=pd.concat([df_x,num_normal],axis=1)
print(df_x)
# #归一化------
num_clients = 5


#打乱数据随机
#
from sklearn.utils import shuffle
train_set = shuffle(train_set,random_state=4)
result = np.array_split(train_set, 5) 
train_noniid=[]
for part in result:
    train_noniid.append(part.values)
#这里是输出taget在每个数据集里面有多少
# print(train_noniid[1])
count=[0,0,0,0,0]
for i in range(len(train_noniid[0])):
    if train_noniid[0][i][14]==1:
        count[0]+=1
    if train_noniid[1][i][14]==1:
        count[1]+=1
    if train_noniid[2][i][14]==1:
        count[2]+=1
    if train_noniid[3][i][14]==1:
        count[3]+=1
    if train_noniid[4][i][14]==1:
        count[4]+=1
print('每组分到数据',count)

#把数据分给5个人
data = []
for subset in train_noniid:
    df = pd.DataFrame(subset)
    df.columns = [*df.columns[:-1], 'target']
    train_label = df['target'].values
    train_data=df.drop('target', axis=1).values
    data.append((train_data, train_label))
test_label=test_set['target'].values
test_data=test_set.drop('target', axis=1).values
data.append((test_data,test_label))

# print(len(data))
# print(len(data[0]))
# print(len(data[0][0]))
# print(len(data[0][0][0]))


lr = 0.001
fl_param = {
    'output_size': 3,
    'client_num': num_clients,
    'model': MLP,
    'data': data,
    'lr': lr,
    'epoch': 3,
    'C': 1,#c是[0,1]代表取百分之多少的样本
    'sigma': 0.5,
    'clip': 2,
    'batch_size': 128,
    'device': device,
}
import warnings
warnings.filterwarnings("ignore")
fl_entity = FedAvgServer(fl_param).to(device)

for e in range(50):
    fl_entity.set_lr(lr)
    acc = fl_entity.global_update()
    print("global epochs = {:d}, acc = {:.4f}".format(e+1, acc))


                   