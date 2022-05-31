import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
# from torchsummary import summary
import numpy as np
from torchvision import models
import matplotlib.pyplot as plt
# import cv2
# import math
# from nobn_cnn import N_BN_CNN
# from small_cnn import CNN
# Device configuration
from scipy.spatial import distance
import pandas as pd
import pylab

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')


# Hyper-parameters
num_epochs = 100
batch_size = 64
learning_rate = 0.001

# Image preprocessing modules
transform = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor()
])


train_dataset = torchvision.datasets.CIFAR10(root='../../data',
                                             train=True,
                                             transform=transform,
                                             download=True)
test_dataset = torchvision.datasets.CIFAR10(root='../../data',
                                            train=False,
                                            transform=transforms.ToTensor(),
                                            download=True)
print(len(test_dataset))

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

def conv3x3(in_channels, out_channels, strides=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=strides, padding=1, bias=False)

class ResidualBlock(nn.Module):
    def __init__(self,in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv = conv3x3(3, 64)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 64, layers[0])
        self.layer2 = self.make_layer(block, 128, layers[1], 2)
        self.layer3 = self.make_layer(block, 256, layers[2], 2)
        self.layer4 = self.make_layer(block, 512, layers[3], 2)
        self.avg_pool = nn.AvgPool2d(4)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self,block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, strides = stride),
                nn.BatchNorm2d(out_channels))

        layers=[]
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels,out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0),-1)
        out = self.fc(out)
        return out



def softmax(x):
    x_row_max = x.max(axis=-1)
    x_row_max = x_row_max.reshape(list(x.shape)[:-1]+[1])
    x = x - x_row_max
    x_exp = np.exp(x)
    x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(x.shape)[:-1]+[1])
    softmax = x_exp / x_exp_row_sum
    return softmax

def get_filter_similar(weight_torch, compress_rate, distance_rate, length, dist_type_1 = 'L2',dist_type="cos"):
        codebook = np.ones(length)
        if len(weight_torch.size()) == 4:
            filter_pruned_num = int(weight_torch.size()[0] * (1 - compress_rate))
            similar_pruned_num = int(weight_torch.size()[0] * distance_rate)
            weight_vec = weight_torch.view(weight_torch.size()[0], -1)

            if dist_type_1 == "l2":
                norm = torch.norm(weight_vec, 2, 1)
                norm_np = norm.cpu().detach().numpy()
            elif dist_type_1 == "l1":
                norm = torch.norm(weight_vec, 1, 1)
                norm_np = norm.cpu().detach().numpy()
            filter_small_index = []
            filter_large_index = []
            filter_large_index = norm_np.argsort()[filter_pruned_num:]  # 从小到大排序，取其对应的索引
            filter_small_index = norm_np.argsort()[:filter_pruned_num]
            # print('filter_small_index:',filter_small_index)
            # # distance using pytorch function
            # similar_matrix = torch.zeros((len(filter_large_index), len(filter_large_index)))
            # for x1, x2 in enumerate(filter_large_index):
            #     for y1, y2 in enumerate(filter_large_index):
            #         # cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
            #         # similar_matrix[x1, y1] = cos(weight_vec[x2].view(1, -1), weight_vec[y2].view(1, -1))[0]
            #         pdist = torch.nn.PairwiseDistance(p=2)
            #         similar_matrix[x1, y1] = pdist(weight_vec[x2].view(1, -1), weight_vec[y2].view(1, -1))[0][0]
            # # more similar with other filter indicates large in the sum of row
            # similar_sum = torch.sum(torch.abs(similar_matrix), 0).numpy()

            # distance using numpy function
            # indices = torch.LongTensor(filter_large_index).cuda()
            # weight_vec_after_norm = torch.index_select(weight_vec, 0, indices).cpu().detach().numpy()
            # for euclidean distance
            weight_vec_after_norm = weight_vec.cpu().detach().numpy()
            if dist_type == "euclidean":
                similar_matrix = distance.cdist(weight_vec_after_norm, weight_vec_after_norm, 'euclidean')
            elif dist_type == "cos":  # for cos similarity
                similar_matrix = 1 - distance.cdist(weight_vec_after_norm, weight_vec_after_norm, 'cosine')
            similar_sum = np.sum(np.abs(similar_matrix), axis=0)

            # for distance similar: get the filter index with largest similarity == small distance
            similar_large_index = similar_sum.argsort()[similar_pruned_num:]
            similar_small_index = similar_sum.argsort()[:  similar_pruned_num]
            count = 0
            for i in range(len(similar_small_index)):
                if similar_small_index[i] in filter_small_index:
                    count = count+1

            similar_index_for_filter = similar_small_index  #[filter_large_index[i] for i in similar_small_index]

            # print('filter_large_index', filter_large_index)
            print('L-filter_small_index', filter_small_index)
            # print('similar_sum', similar_sum)
            # print('similar_large_index', similar_large_index)
            print('_GMorCOS_similar_small_index', similar_small_index)
            print('{}_and_{},simliar_number:{}%'.format(dist_type_1,dist_type,count/filter_pruned_num))
            print('filter_pruned_num',filter_pruned_num)
            # print('similar_index_for_filter', similar_index_for_filter)
            kernel_length = weight_torch.size()[1] * weight_torch.size()[2] * weight_torch.size()[3]
            for x in range(0, len(similar_index_for_filter)):
                codebook[
                similar_index_for_filter[x] * kernel_length: (similar_index_for_filter[x] + 1) * kernel_length] = 0
            # print("similar index done")
        else:
            pass
        return codebook,filter_small_index


def get_filter_similar_with_group(weight_torch, compress_rate, distance_rate, length, dist_type="cos"):
    codebook = np.ones(length)
    if len(weight_torch.size()) == 4:
        filter_pruned_num = int(weight_torch.size()[0] * (1 - compress_rate))
        # similar_pruned_num = int(weight_torch.size()[0] * distance_rate)
        weight_vec = weight_torch.view(weight_torch.size()[0], -1)

        # if dist_type == "l2" or "cos":
        #     norm = torch.norm(weight_vec, 2, 1)
        #     norm_np = norm.cpu().detach().numpy()
        # elif dist_type == "l1":
        #     norm = torch.norm(weight_vec, 1, 1)
        #     norm_np = norm.detach().numpy()
        filter_small_index = []
        filter_large_index = []
        # filter_large_index = norm_np.argsort()[filter_pruned_num:]  # 从小到大排序，取其对应的索引
        # filter_small_index = norm_np.argsort()[:filter_pruned_num]

        # # distance using pytorch function
        # similar_matrix = torch.zeros((len(filter_large_index), len(filter_large_index)))
        # for x1, x2 in enumerate(filter_large_index):
        #     for y1, y2 in enumerate(filter_large_index):
        #         # cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        #         # similar_matrix[x1, y1] = cos(weight_vec[x2].view(1, -1), weight_vec[y2].view(1, -1))[0]
        #         pdist = torch.nn.PairwiseDistance(p=2)
        #         similar_matrix[x1, y1] = pdist(weight_vec[x2].view(1, -1), weight_vec[y2].view(1, -1))[0][0]
        # # more similar with other filter indicates large in the sum of row
        # similar_sum = torch.sum(torch.abs(similar_matrix), 0).numpy()

        # distance using numpy function
        # indices = torch.LongTensor(filter_large_index).cuda()
        # weight_vec_after_norm = torch.index_select(weight_vec, 0, indices).cpu().detach().numpy()
        # for euclidean distance
        similar_pruned_num = int(32*distance_rate)
        weight_vec_after_norm = weight_vec.cpu().detach().numpy()
        # np.random.shuffle(weight_vec_after_norm)
        similar_index_for_filter=[]
        for i in range(0,weight_vec.size()[0],32):
            if dist_type == "euclidean":
                similar_matrix = distance.cdist(weight_vec_after_norm[i:i+32,:], weight_vec_after_norm[i:i+32, :], 'euclidean')
            elif dist_type == "cos":  # for cos similarity
                similar_matrix = 1 - distance.cdist(weight_vec_after_norm, weight_vec_after_norm, 'cosine')
            similar_sum = np.sum(np.abs(similar_matrix), axis=0)

            # for distance similar: get the filter index with largest similarity == small distance
            similar_large_index = similar_sum.argsort()[similar_pruned_num:]
            similar_small_index = similar_sum.argsort()[:  similar_pruned_num]
            similar_small_index = similar_small_index+i
            similar_index_for_filter.append(similar_small_index)  # [filter_large_index[i] for i in similar_small_index]
        total=[]
        for i in similar_index_for_filter:
            total +=list(i)
        # print('filter_large_index', filter_large_index)
        # print('filter_small_index', filter_small_index)
        # print('similar_sum', similar_sum)
        # print('similar_large_index', similar_large_index)
        # print('similar_small_index', similar_small_index)
        # print('similar_index_for_filter', similar_index_for_filter)
        print('group_total_filter', total)
        kernel_length = weight_torch.size()[1] * weight_torch.size()[2] * weight_torch.size()[3]
        # for x in range(0, len(similar_index_for_filter)):
        #     codebook[
        #     similar_index_for_filter[x] * kernel_length: (similar_index_for_filter[x] + 1) * kernel_length] = 0
        # print("similar index done")
    else:
        pass
    return total

def self_attention_torch(weight):
    weight_vec = weight.view([weight.shape[0], -1])
    # c = np.reshape(weight,[weight.shape[0],-1])
    weight_vec_tran = weight.view([weight.shape[0], -1])
    weight_vec_tran = weight_vec_tran.transpose(0, 1)
    print(weight_vec.shape, weight_vec_tran.shape)
    # d = np.matmul(b, c.T)
    weight_matrix = torch.matmul(weight_vec, weight_vec_tran)
    print('re', weight_matrix)
    weight_attention = torch.nn.functional.softmax(weight_matrix, dim=-1)
    weight_diagonal = np.diagonal(weight_attention.cpu().detach().numpy())
    print(weight_diagonal, weight_diagonal.shape)
    weight_diag_sums = np.sum(weight_diagonal)
    print('total_sum', weight_diag_sums)

    return weight_attention, weight_diagonal, weight_diag_sums

def self_attention_np(weight):
    weight_vec = np.reshape(weight, [weight.shape[0], -1])
    # weight_vec_tran = np.reshape(weight, [weight.shape[0], -1])
    print(weight_vec.shape)
    weight_matrix = np.matmul(weight_vec, weight_vec.T)
    print('re', weight_matrix)
    weight_attention = softmax(weight_matrix)
    weight_diagonal = np.diagonal(weight_attention)
    print(weight_diagonal, weight_diagonal.shape)
    weight_diag_sums = np.sum(weight_diagonal)
    # # print('attention', m, np.max(m, axis=1))
    # print('sum', np.sum(m))
    print('total_sum', weight_diag_sums)

    return weight_attention, weight_diagonal, weight_diag_sums


def compute_weight_hist(weight):
    weight_vec = weight.view([weight.shape[0], -1]).cpu().detach().numpy()
    # hist,bins = np.histogram(weight_vec,bins=50)
    # hist = hist / hist.max()
    # print(hist, bins)
    plt.hist(weight_vec, bins='auto')
    plt.show()
    # hist = hist/hist.max()
    # print(hist,bins)
    # pylab.hist(hist, bins,range=[-0.5, 0.5]) #normed=1
    # pylab.show()
    return None

def compute_entropy_information(weight):
    w,d,s = self_attention_torch(weight)
    attention = d
    print(attention)
    log_a = np.log2(attention)
    d = np.multiply(attention,log_a)
    result = -np.sum(d)
    c = len(d)
    print(c)
    max = -np.log2(1/c)
    print(max)
    result = result/max
    return result

if __name__ == '__main__':

    # model = ResNet(ResidualBlock, [2,2,2,2])
    # model.load_state_dict(torch.load('E:\pycharm\FrequencyResearch\\resnet18_test.ckpt'))
    model = models.resnet152(pretrained=True)
    # model.load_state_dict(torch.load('E:\pycharm\FrequencyResearch\\resnet18_test.ckpt'))
    model.cuda()
    # summary(model, (3, 32, 32))
    # parm = {}
    # for name, parameters in model.named_parameters():
    #     print(name, ':', parameters.size())
    #     parm[name] = parameters.detach().numpy()
    # weight = parm['features.8.weight']
    # print(weight,weight.shape)
    # b = np.transpose(weight,[1,0, 2, 3])
    # b = np.reshape(weight,[weight.shape[0], -1])
    # c = np.reshape(weight,[weight.shape[0],-1])
    # print(b.shape,c.shape)
    # d = np.matmul(b, c.T)
    # print('re',d)
    # m = softmax(d)
    # feature = np.diagonal(m)
    # print(feature,feature.shape)
    # sums =np.sum(feature)
    # # # print('attention', m, np.max(m, axis=1))
    # # print('sum', np.sum(m))
    # print('total_sum',sums)
    # weight = np.reshape(weight, [weight.shape[0],weight.shape[1],-1])
    # plt.figure('weight')
    # plt.xlim((-15, 15))
    # plt.ylim((-2, 2))
    # ax = plt.gca()
    # ax.spines['right'].set_color('none')
    # ax.spines['top'].set_color('none')
    # ax.xaxis.set_ticks_position('bottom')
    # ax.yaxis.set_ticks_position('left')
    # ax.spines['bottom'].set_position(('data', 0))
    # ax.spines['left'].set_position(('data', 0))
    # x = np.linspace(-5, 5, 9)
    # for i in range(0,weight.shape[0],64):
    #     for j in range(0,weight.shape[1],64):
    #         plt.plot(x, weight[i, j, :])
    # plt.show()
    #torch实现
    # parm = {}
    # for name, parameters in model.named_parameters():
    #     print(name, ':', parameters.size())
    #     parm[name] = parameters
    # weight = parm['features.denseblock4.denselayer16.conv1.weight']
    # print(weight,weight.shape)
    # # b = np.transpose(weight,[1,0, 2, 3])
    # # b = np.reshape(weight,[weight.shape[0], -1])
    # b = weight.view([weight.shape[0],-1])
    # # c = np.reshape(weight,[weight.shape[0],-1])
    # c = weight.view([weight.shape[0], -1])
    # c = c.transpose(0,1)
    # print(b.shape,c.shape)
    # # d = np.matmul(b, c.T)
    # d= torch.matmul(b, c)
    # print('re',d)
    # m = torch.nn.functional.softmax(d,dim=-1)
    # feature = np.diagonal(m.detach().numpy())
    # print(feature,feature.shape)
    # sums =np.sum(feature)
    # print('total_sum',sums)
    parm = {}
    for name, parameters in model.named_parameters():
        print(name, ':', parameters.size())
        parm[name] = parameters
    weight = parm['conv1.weight']
    print(weight,weight.shape)
    compress_rate = 0.9
    filter_pruned_num = int(weight.size()[0] * (1 - compress_rate))
    type1 = 'l1'
    type2 = 'euclidean'
    #GM COS l1l2
    # _,filter = get_filter_similar(weight,compress_rate=0.9,distance_rate=0.1, length=512,dist_type_1= type1,dist_type=type2)
    # total = get_filter_similar_with_group(weight,compress_rate=0.9,distance_rate=0.1, length=512,dist_type='euclidean')
    # sim = [x for x in filter if x in total]
    # print('similar number% with {} and GGM!!!'.format(type1), len(sim)/filter_pruned_num)

    #self attention


    w,s,t = self_attention_torch(weight)
    attention_filter = s.argsort()
    print("attention",attention_filter)
    # sim1 = [x for x in attention_filter if x in filter]
    # print('similar number with {} and attention!!!'.format(type1), len(sim1)/filter_pruned_num)
    # print(codebook)
    plt.matshow(w.cpu().detach().numpy())
    plt.show()

    #协方差矩阵
    # weight_vec = weight.view([weight.shape[0], -1])
    # w = weight_vec.cpu().detach().numpy()
    # cov = np.cov(w)
    #
    # cov_d = np.diagonal(cov)
    # cov_d_filter = cov_d.argsort()
    # print(cov_d_filter)
    # sim2 = [x for x in cov_d_filter if x in filter]
    # print('similar number with  {}  and cov!!!'.format(type1), len(sim2)/filter_pruned_num)
    # plt.matshow(cov)
    # plt.show()
    #计算直方图
    compute_weight_hist(weight)

    # #计算attention信息熵
    # result = compute_entropy_information(weight)
    # print(result)




















