
from __future__ import print_function
import argparse
import torch
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
from dataset import ShapeNetDataset
from model import PointNetCls
import torch.nn.functional as F


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='', help='model path')
    parser.add_argument('--num_points', type=int, default=2500, help='input batch size')
    parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
    parser.add_argument('--feature_transform', default=False, help="use feature transform")

    opt = parser.parse_args()
    print(opt)

    test_dataset = ShapeNetDataset(
        root='../shapenet_data/shapenetcore_partanno_segmentation_benchmark_v0',
        classification=True,
        split='test',
        data_augmentation=False)

    train_dataset = ShapeNetDataset(
        root='../shapenet_data/shapenetcore_partanno_segmentation_benchmark_v0',
        classification=True,
        split='train',
        data_augmentation=False)

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batchSize,
        shuffle=True)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batchSize,
        shuffle=True)

    classifier = PointNetCls(num_classes=len(test_dataset.classes), feature_transform=opt.feature_transform)
    classifier.cuda()

    classifier.load_state_dict(torch.load(opt.model))
    classifier.eval()

    data_len = len(train_dataloader)

    train_accuracy = 0.0

    for i, data in enumerate(train_dataloader, 0):
        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        pred, trans, trans_feat = classifier(points)
        loss = F.nll_loss(pred, target)
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        print('[%d/%d] train loss: %f accuracy: %f' % (i+1, data_len, loss.item(), correct.item() / float(opt.batchSize)))
        train_accuracy += correct.item() / float(opt.batchSize)

    train_accuracy /= i

    data_len = len(test_dataloader)
    test_accuracy = 0.0

    for i, data in enumerate(test_dataloader, 0):
        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        classifier = classifier.eval()
        pred, _, _ = classifier(points)
        loss = F.nll_loss(pred, target)
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        print('[%d/%d] test loss: %f accuracy: %f' % (i+1, data_len, loss.item(), correct.item() / float(opt.batchSize)))
        test_accuracy += correct.item() / float(opt.batchSize)

    test_accuracy /= i

    print('Average Train Accuracy: %f Average Test Accuracy: %f' % (train_accuracy, test_accuracy))



if __name__ == '__main__':
    main()
