
from __future__ import print_function

import copy

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


class TNet(nn.Module):
    def __init__(self, k=64):
        super(TNet, self).__init__()

        self.k = k

        # Each layer has batchnorm and relu on it
        #TODO
        # layer 1: k -> 64
        self.conv1 = nn.Sequential(nn.Conv1d(k, 64, 1), nn.BatchNorm1d(64),
                                   nn.ReLU(inplace=True))
        #TODO
        # layer 2:  64 -> 128
        self.conv2 = nn.Sequential(nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128),
                                   nn.ReLU(inplace=True))
        #TODO
        # layer 3: 128 -> 1024
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024),
                                   nn.ReLU(inplace=True))
        self.mpool = nn.Sequential(nn.AdaptiveMaxPool1d(1))
        #TODO
        # fc 1024 -> 512
        self.fc1 = nn.Sequential(nn.Linear(1024, 512), nn.BatchNorm1d(512),
                                 nn.ReLU(inplace=True))
        #TODO
        # fc 512 -> 256
        self.fc2 = nn.Sequential(nn.Linear(512, 256), nn.BatchNorm1d(256),
                                 nn.ReLU(inplace=True))
        #TODO
        # fc 256 -> k*k (no batchnorm, no relu)
        self.fc3 = nn.Linear(256, k * k)
        #TODO
        # ReLU activationfunction
        self.fc3.bias = torch.nn.Parameter(torch.eye(k).view(-1))



    def forward(self, x):
        batch_size, _, num_points = x.shape
        batch_size = x.size()[0]
        #TODO
        # apply layer 1
        x = self.conv1(x)
        #TODO
        # apply layer 2
        x = self.conv2(x)
        #TODO
        # apply layer 3
        x = self.conv3(x)
        #TODO
        # do maxpooling and flatten
        x = self.mpool(x)
        x = x.view(x.shape[:-1])
        #TODO
        # apply fc layer 1
        x = self.fc1(x)
        #TODO
        # apply fc layer 2
        x = self.fc2(x)
        #TODO
        # apply fc layer 3
        x = self.fc3(x)
        #TODO
        #reshape output to a b*k*k tensor
        x = x.view(batch_size, self.k, self.k)
        #TODO
        # define an identity matrix to add to the output. This will help with the stability of the results since we want our transformations to be close to identity


        #TODO
        # return output
        return x


class PointNetfeat(nn.Module):
    def __init__(self, global_feat = True, feature_transform = False):
        super(PointNetfeat, self).__init__()
        self.feature_transform= feature_transform
        self.global_feat = global_feat
        self.tnet1 = TNet(k=3)
        #TODO
        # Use TNet to apply transformation on input and multiply the input points with the transformation

        #TODO
        # layer 1:3 -> 64
        self.conv1 = nn.Sequential(nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64),
                                   nn.ReLU(inplace=True))     
        #TODO
        # Use TNet to apply transformation on features and multiply the input features with the transformation 
        #                                                                        (if feature_transform is true)

        #TODO
        # layer2: 64 -> 128
        self.conv2 = nn.Sequential(nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128),
                                   nn.ReLU(inplace=True))
        #TODO
        # layer 3: 128 -> 1024 (no relu)
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024))
        #TODO
        # ReLU activation
        self.mpool = nn.Sequential(nn.AdaptiveMaxPool1d(1))

        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.tnet2 = TNet(k=64)



    def forward(self, x):
        batch_size, _, num_points = x.shape
        n_pts = x.size()[2]
        #TODO
        # input transformation, you will need to return the transformation matrix as you will need it for the regularization loss
        trans = self.tnet1(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        
        #TODO
        # apply layer 1
        x = self.conv1(x)
        #TODO
        # feature transformation, you will need to return the transformation matrix as you will need it for the regularization loss
        if self.feature_transform:
            trans_feat = self.tnet2(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        #TODO
        # apply layer 2
        x = self.conv2(x)
        #TODO
        # apply layer 3
        x = self.conv3(x)
        #TODO
        # apply maxpooling
        x = self.mpool(x)
        x = x.view(x.shape[:-1])

        #TODO
        # return output, input transformation matrix, feature transformation matrix
        if self.global_feat: # This shows if we're doing classification or segmentation

            #if self.feature_transform:
            return x, trans, trans_feat

        else:
            
            #if self.feature_transform:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


class PointNetCls(nn.Module):
    def __init__(self, num_classes = 2, feature_transform=False):
        super(PointNetCls, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1), trans, trans_feat


class PointNetDenseCls(nn.Module):
    def __init__(self, num_classes = 2, feature_transform=False):
        super(PointNetDenseCls, self).__init__()
        #TODO
        # get global features + point features from PointNetfeat
        self.k = num_classes
        self.feature_transform = feature_transform
        self.pointNetFeat = PointNetfeat(global_feat=False, feature_transform=feature_transform)

        #TODO
        # layer 1: 1088 -> 512
        self.conv1 = nn.Sequential(nn.Conv1d(1088, 512, 1), nn.BatchNorm1d(512),
                                   nn.ReLU(inplace=True))
        #TODO
        # layer 2: 512 -> 256
        self.conv2 = nn.Sequential(nn.Conv1d(512, 256, 1), nn.BatchNorm1d(256),
                                   nn.ReLU(inplace=True))
        #TODO
        # layer 3: 256 -> 128
        self.conv3 = nn.Sequential(nn.Conv1d(256, 128, 1), nn.BatchNorm1d(128),
                                   nn.ReLU(inplace=True))
        #TODO
        # layer 4:  128 -> k (no ru and batch norm)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        #TODO
        # ReLU activation


    
    def forward(self, x):
        #TODO
        # You will need these extra outputs: 
        # trans = output of applying TNet function to input
        # trans_feat = output of applying TNet function to features (if feature_transform is true)
        # (you can directly get them from PointNetfeat)

        batch_size = x.size()[0]
        n_pts = x.size()[2]
        x, trans, trans_feat = self.pointNetFeat(x)
        #TODO
        # apply layer 1
        x = self.conv1(x)
        #TODO
        # apply layer 2
        x = self.conv2(x)
        #TODO
        # apply layer 3
        x = self.conv3(x)
        #TODO
        # apply layer 4
        x = self.conv4(x)
        #TODO
        # apply log-softmax
        x = x.transpose(2, 1)
        x = x.reshape(-1, self.k)
        x = F.log_softmax(x, dim=-1)
        x = x.view(batch_size, n_pts, self.k)
        return x, trans, trans_feat


def feature_transform_regularizer(trans):

    trans=trans.cuda()
    #batch_size, feature_size, _ = trans.shape
    #TODO
    # compute I - AA^t
    I_matrix = torch.eye(trans.size()[1])[None, :, :]
    AAT = torch.bmm(trans, trans.transpose(2, 1))
    diffMat = AAT - I_matrix.cuda()
    #TODO
    # compute norm
    loss = torch.norm(diffMat, dim=(1, 2))
    #TODO
    # compute mean norms and return
    loss = torch.mean(loss)
    return loss


if __name__ == '__main__':
    sim_data = Variable(torch.rand(32,3,2500))
    trans = TNet(k=3)
    out = trans(sim_data)
    print('TNet', out.size())
    print('loss', feature_transform_regularizer(out.cuda()))

    sim_data_64d = Variable(torch.rand(32, 64, 2500))
    trans = TNet(k=64)
    out = trans(sim_data_64d)
    print('TNet 64d', out.size())
    print('loss', feature_transform_regularizer(out.cuda()))

    pointfeat = PointNetfeat(global_feat=True)
    out, _, _ = pointfeat(sim_data)
    print('global feat', out.size())

    pointfeat = PointNetfeat(global_feat=False)
    out, _, _ = pointfeat(sim_data)
    print('point feat', out.size())

    cls = PointNetCls(num_classes = 5)
    out, _, _ = cls(sim_data)
    print('class', out.size())

    seg = PointNetDenseCls(num_classes = 3)
    out, _, _ = seg(sim_data)
    print('seg', out.size())
    

