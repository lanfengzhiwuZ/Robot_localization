from data_utils.ModelNetDataLoader import ModelNetDataLoader
import argparse
import numpy as np
import os
import torch
import logging

import sys
import importlib
import matplotlib.pyplot as plt
 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
 
def pc_normalize(pc):  #点云数据归一化
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())
def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Testing')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=12, help='batch size in training')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--num_point', type=int, default=10000, help='Point Number')
    parser.add_argument('--log_dir', type=str, default='pointnet2_cls_msg', help='Experiment root')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--num_votes', type=int, default=3, help='Aggregate classification scores with voting')
    return parser.parse_args()
#加载数据集
dataset='/Pointnet/evalset/aaa_1.txt'
pcdataset = np.loadtxt(dataset, delimiter=',').astype(np.float32)#数据读取，我的数据是三个维度，数据之间是空格，如果是逗号修改一下即可
point_set = pcdataset[0:10000, :] #我的输入数据设置为原始数据中10000个点
point_set[:, 0:3] = pc_normalize(point_set[:, 0:3]) #归一化数据
point_set = point_set[:, 0:3] 
point_set = point_set.transpose(1,0)#将数据由N*C转换为C*N
# print(point_set.shape)
point_set = point_set.reshape(1, 3, 10000)
n_points = point_set
point_set = torch.as_tensor(point_set)#需要将数据格式变为张量，不然会报错
point_set = point_set.cuda()
# print(point_set.shape)
# print(point_set.shape)
#分类测试函数
def test(model,point_set, num_class=40, vote_num=1):
    # mean_correct = []
    classifier = model.eval()
    class_acc = np.zeros((num_class, 3))
    vote_pool = torch.zeros(1, 40).cuda()
    for _ in range(vote_num):
        pred, _ = classifier(point_set)
        print(pred)
        vote_pool += pred
    pred = vote_pool / vote_num
    # 对预测结果每行取最大值得到分类
    pred_choice = pred.data.max(1)[1]
    print(pred_choice)
    #可视化
    file_dir = '/Pointnet/visualizer'
    save_name_prefix = 'pred'
    draw(n_points[:, 0, :], n_points[:, 1, :], n_points[:, 2, :], save_name_prefix, file_dir, color=pred_choice)
    return pred_choice
#定义可视化函数
def draw(x, y, z, name, file_dir, color=None):
    """
    绘制单个样本的三维点图
    """
    if color is None:
        for i in range(len(x)):
            ax = plt.subplot(projection='3d')  # 创建一个三维的绘图工程
            save_name = name + '-{}.png'.format(i)
            save_name = os.path.join(file_dir,save_name)
            ax.scatter(x[i], y[i], z[i],s=0.1, c='r')
            ax.set_zlabel('Z')  # 坐标轴
            ax.set_ylabel('Y')
            ax.set_xlabel('X')
            plt.draw()
            plt.savefig(save_name)
            plt.show()
    else:
        colors = ['red', 'blue', 'green', 'yellow', 'orange', 'tan', 'orangered', 'lightgreen', 'coral', 'aqua']
        for i in range(len(x)):
            ax = plt.subplot(projection='3d')  # 创建一个三维的绘图工程
            save_name = name + '-{}-{}.png'.format(i, color[i])
            save_name = os.path.join(file_dir, save_name)
            ax.scatter(x[i], y[i], z[i], s=0.1, c=colors[color[i]])
            ax.set_zlabel('Z')  # 坐标轴
            ax.set_ylabel('Y')
            ax.set_xlabel('X')
            plt.draw()
            plt.savefig(save_name)
            plt.show()
 
def main(args):
    def log_string(str):
        logger.info(str)
        print(str)
    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    '''CREATE DIR'''
    experiment_dir = '/Pointnet/log/classification/pointnet2_ssg_wo_normals/'
    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    '''
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)
    '''
    num_class = args.num_category
    #选择模型
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    model = importlib.import_module(model_name)
    
    classifier = model.get_model(num_class, normal_channel=args.use_normals)
    if not args.use_cpu:
        classifier = classifier.cuda()
    #选择训练好的.pth文件
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])
    #预测分类
    with torch.no_grad():
         pred_choice = test(classifier.eval(), point_set, vote_num=args.num_votes, num_class=num_class)
         #log_string('pred_choice: %f' % (pred_choice))
 
if __name__ == '__main__':
    args = parse_args()
    main(args)