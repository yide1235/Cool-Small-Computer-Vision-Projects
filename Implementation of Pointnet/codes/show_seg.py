from __future__ import print_function
from show3d_balls import showpoints
import argparse
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
from dataset import ShapeNetDataset
from model import PointNetDenseCls
import matplotlib.pyplot as plt


#showpoints(np.random.randn(2500,3),c1 = np.random.uniform(0,1,size = (2500)))

parser = argparse.ArgumentParser()

#parser.add_argument('--model', type=str, default='../pretrained_networks/latest_segmentation.pt', help='model path')
parser.add_argument('--model', type=str, default='./seg/seg_model_Airplane_9.pth', help='model path')
parser.add_argument('--idx', type=int, default=1, help='model index')
parser.add_argument('--dataset', type=str, default='../shapenet_data/shapenetcore_partanno_segmentation_benchmark_v0', help='dataset path')
parser.add_argument('--class_choice', type=str, default='Chair', help='class choice')
parser.add_argument('--feature_transform', default=False, help="use feature transform")

opt = parser.parse_args()
print(opt)

d = ShapeNetDataset(
    root=opt.dataset,
    class_choice=[opt.class_choice],
    split='test',
    data_augmentation=False)

idx = opt.idx

print("model %d/%d" % (idx, len(d)))
point, seg = d[idx]
print(point.size(), seg.size())
point_np = point.numpy()

cmap = plt.cm.get_cmap("hsv", 10)
cmap = np.array([cmap(i) for i in range(10)])[:, :3]
gt = cmap[seg.numpy() - 1, :]

state_dict = torch.load(opt.model)
classifier = PointNetDenseCls(num_classes=state_dict['conv4.weight'].size()[0], feature_transform=opt.feature_transform)
classifier.load_state_dict(state_dict)
classifier.eval()

point = point.transpose(1, 0).contiguous()

point = Variable(point.view(1, point.size()[0], point.size()[1]))
pred, _, _ = classifier(point)
pred_choice = pred.data.max(2)[1]
print(pred_choice)

pred_color = cmap[pred_choice.numpy()[0], :]

showpoints(point_np, gt, pred_color)




