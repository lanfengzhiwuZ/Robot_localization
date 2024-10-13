import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd


#读取路径
data = pd.read_csv("lianzhuquB_combine.csv")


# 查看数组的维度  
print("Shape:", data.shape)  
row,columns=data.shape
data = data.values
print(data[0])
print(data[1])
print(data[2])
data = torch.Tensor(data)  # 将数据转换为PyTorch张量
print(data[0])


# 第一个编码器和解码器将输入的10维数据编码为4维潜在表示，
# 第二个编码器和解码器将这个4维表示进一步编码为2维表示，
# 最后，第二个解码器将2维表示解码为10维表示。
lay1 = columns
lay2 = columns+50
lay3 = columns+100
class StackedAutoencoder(nn.Module):
    def __init__(self):
        super(StackedAutoencoder, self).__init__()

        self.encoder1 = nn.Sequential(
            nn.Linear(lay1, lay2),
            nn.ReLU()
        )

        self.encoder2 = nn.Sequential(
            nn.Linear(lay2, lay3),
            nn.ReLU()
        )

        self.decoder1 = nn.Sequential(
            nn.Linear(lay3, lay2),
            nn.ReLU()
        )

        self.decoder2 = nn.Sequential(
            nn.Linear(lay2, lay1),
            nn.ReLU()
        )
        self.dropout = torch.nn.Dropout(0.5)
# 你的前向传播定义  
    def forward(self, x):
        x = self.encoder1(x)
        x = self.dropout(x)
        x = self.encoder2(x)
        x = self.dropout(x)
        x = self.decoder1(x)
        x = self.dropout(x)
        x = self.decoder2(x)
        return x


# 初始化模型  
model = StackedAutoencoder()
# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)#学习率
epoch_Max = 1200

for epoch in range(epoch_Max):
    output = model(data)
    loss = criterion(output, data)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("Epoch [{}/{}], Loss: {:.4f}".format(epoch+1, epoch_Max, loss.item()))
    



# 原始数据输入到训练好的模型中，并获取重建的数据
reconstructed_data = model(data)


# 将重建数据与原始数据进行比较，以评估模型的性能，用于计算平均重建误差
mse_loss = nn.MSELoss()
reconstruction_error = mse_loss(data, reconstructed_data).item()
print("Average Reconstruction Error: {:.4f}".format(reconstruction_error))

print(reconstructed_data)

# 将数据转换为DataFrame  
ndarray =reconstructed_data.detach().numpy()
reconstructed_data_np = np.array(ndarray)
data_df = pd.DataFrame(reconstructed_data_np)  
  
# 将DataFrame保存为CSV文件  
filename = 'lianzhuquB_combine_reconstructed6.csv'  # 指定文件名  
data_df.to_csv(filename, index=False)  # 保存为CSV文件，不包括行索引


# 可视化工具如matplotlib将原始数据和重建数据进行可视化比较
fig, ax = plt.subplots(2, 10, figsize=(40, 4)) 

for i in range(10):
    ax[0][i].imshow(data[i].reshape(1, columns), cmap='gray')
    ax[0][i].axis('off')
    ax[1][i].imshow(reconstructed_data[i].detach().numpy().reshape(1, columns), cmap='gray')
    ax[1][i].axis('off')


plt.show()


