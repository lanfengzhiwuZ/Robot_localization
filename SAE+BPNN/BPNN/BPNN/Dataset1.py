#库的导入
import os
import csv
import numpy as np
from numpy import nan
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import preprocessing
from sklearn.model_selection import train_test_split #训练集，测试集划分函数
from sklearn.utils import Bunch
import torch
import torch.nn.functional as Fun


#设置超参数
lr=0.02 #学习率
epochs=300 #训练轮数
n_feature=14 #输入特征
n_hidden1=20 #隐层节点数
n_hidden2=12
n_output=4 #输出(分类的数量)


#1.准备数据

with open('./ALL.csv', 'r', encoding='utf-8') as csvFile:  
    csv_data = csv.reader(csvFile)
    area =np.array([i for i in csv_data])

row,columns= area.shape##数据的行数和列数
print("行数:", row)  
print("列数:", columns)

#csvFile = open("./ALL.csv","r")
#csv_data = csv.reader(csvFile)
#area =np.array([i for i in csv_data])

feature_names=area[0,:columns-1]
da=area[1:,:columns-1]


##print(da)
data=[]
##data这个数据中的str转换成float
for i in area[1:,:columns-1]:
    temp=[]
    for j in i:
        if j=='?':
            temp.append(nan)
        else:
            temp.append(float(j))
    data.append(temp)##数据

target=[]
##i是str
for i in area[1:,columns-1]: ##除去第一行的第5列
    if i=='0':
        target.append(0)
    if i=='1':
        target.append(1)
    if i=='2':
        target.append(2)
    if i=='3':
        target.append(3)

target_names=['lianganqu','feiliaoqu','lianzhuquA','lianzhuquB']
print(target)
data = np.array(data)
target = np.array(target)
feature_names = np.array(feature_names)
target_names = np.array(target_names)

##生成数据集
real_data = Bunch(data=data, target=target, feature_names = feature_names, target_names = target_names)
#设置训练集数据80%，测试集20%
x_train0,x_test0,y_train,y_test=train_test_split(real_data.data,real_data.target,test_size=0.2,random_state=22)
##print(real_data)
print(real_data.target)



#归一化(也就是所说的min-max标准化)通过调用sklearn库的标准化函数
min_max_scaler = preprocessing.MinMaxScaler()
x_train = min_max_scaler.fit_transform(x_train0)
x_test = min_max_scaler.fit_transform(x_test0)
 
#将数据类型转换为tensor方便pytorch使用
x_train=torch.FloatTensor(x_train)
y_train=torch.LongTensor(y_train)
x_test=torch.FloatTensor(x_test)
y_test=torch.LongTensor(y_test)
 
#2.定义BP神经网络
class BPNetModel(torch.nn.Module):
    def __init__(self,n_feature,n_hidden1,n_hidden2,n_output):
        super(BPNetModel, self).__init__()
        self.hiddden_1=torch.nn.Linear(n_feature,n_hidden1)#定义隐层网络
        self.hiddden_2=torch.nn.Linear(n_hidden1,n_hidden2)
        self.out=torch.nn.Linear(n_hidden2,n_output)#定义输出层网络
    def forward(self,x):
        x=Fun.relu(self.hiddden_1(x)) #隐层激活函数采用relu()函数
        x=Fun.relu(self.hiddden_2(x))
        out=Fun.softmax(self.out(x),dim=1) #输出层采用softmax函数
        return out

#3.定义优化器和损失函数
net=BPNetModel(n_feature=n_feature,n_hidden1=n_hidden1,n_hidden2=n_hidden2,n_output=n_output) #调用网络
optimizer=torch.optim.Adam(net.parameters(),lr=lr) #使用Adam优化器，并设置学习率
loss_fun=torch.nn.CrossEntropyLoss()  #对于多分类一般使用交叉熵损失函数
 
#4.训练数据
loss_steps=np.zeros(epochs) #构造一个array([0., 0., 0., 0., 0.])里面有epochs个0
accuracy_steps=np.zeros(epochs)
 
for epoch in range(epochs):
    y_pred=net(x_train) #前向传播
    loss=loss_fun(y_pred,y_train)#预测值和真实值对比
    optimizer.zero_grad() #梯度清零
    loss.backward() #反向传播
    optimizer.step() #更新梯度
    loss_steps[epoch]=loss.item()#保存loss
    running_loss = loss.item()
    print(f"第{epoch}次训练\n损失率 {running_loss}".format(epoch,running_loss) )
    with torch.no_grad(): #下面是没有梯度的计算,主要是测试集使用，不需要再计算梯度了
        y_pred=net(x_test)
        correct=(torch.argmax(y_pred,dim=1)==y_test).type(torch.FloatTensor)
        accuracy_steps[epoch]=correct.mean()
        print("预测准确率", accuracy_steps[epoch])
 
#print("测试鸢尾花的预测准确率",accuracy_steps[-1])
 
#5.绘制损失函数和精度
fig_name="area_dataset_classify_BPNet"
fontsize=15
fig,(ax1,ax2)=plt.subplots(2,figsize=(15,12),sharex=True)
ax1.plot(accuracy_steps)
ax1.set_ylabel("test accuracy",fontsize=fontsize)
ax1.set_title(fig_name,fontsize="xx-large")
ax2.plot(loss_steps)
ax2.set_ylabel("train lss",fontsize=fontsize)
ax2.set_xlabel("epochs",fontsize=fontsize)
plt.tight_layout()
plt.savefig(fig_name+'.png')
plt.show()
 
 










