from model import UNet
from dataloader import Cell_data


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torch.optim as optim
import matplotlib.pyplot as plt

import os

#import any other libraries you need below this line
import numpy as np
import PIL
from PIL import Image, ImageOps

transform1=torchvision.transforms.ToPILImage()
from PIL import ImageEnhance



# Paramteres

# learning rate
lr = 0.01
# number of training epochs
epoch_n =600
# input image-mask size
image_size = 572
# root directory of project
root_dir = os.getcwd()
# training batch size
batch_size = 2
# use checkpoint model for training
load = False
# use GPU for training
gpu = True

data_dir = os.path.join(root_dir, 'data/cells')



trainset = Cell_data(data_dir=data_dir, size=image_size)
trainloader = DataLoader(trainset, batch_size=2, shuffle=True)







testset = Cell_data(data_dir=data_dir, size=image_size, train=False)
testloader = DataLoader(testset, batch_size=2,shuffle=True)



"""
a_image,a_label=trainset.__getitem__(4)

a_label=a_label.squeeze()
a_label=a_label.squeeze()
a_label=transform1(a_label)
a_label.show()

i have some problem, the dataloader makes all picture broken, so this is where i tested
"""






device = torch.device('cuda:0' if gpu else 'cpu')




model = UNet().to('cuda:0').to(device)

if load:
    print('loading model')
    model.load_state_dict(torch.load('./checkpoint.pt'))

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=lr,momentum=0.8,weight_decay=0.0005)


train_loss=[]
test_loss=[]



model.train()
for e in range(epoch_n):
    epoch_loss = 0
    model.train()

    

    for i, data in enumerate(trainloader):
        image, label = data
       

        
        #image and label are both 2 1 572 572

       
        image1=image.squeeze()
        image1=image1.squeeze()


        #image2.show()

        #print(label.size())
        label1=label.squeeze()
        label3=label1.squeeze()
        #print(label3.size())
        label2=transform1(label3)
        #label2.show()



        """
        image2=label3[0]
        image_numpy=image2.cpu().numpy()
      
        plt.imshow(image_numpy,cmap='gray')
        plt.show()
        image2=transform1(image1)
        """







        #image = image.unsqueeze(1).to(device)
    
        image=image.to(device)
        label=label.long().to(device)

 
        pred = model(image)

       

        crop_x = (label.shape[1] - pred.shape[2]) // 2
        crop_y = (label.shape[2] - pred.shape[3]) // 2

        label = label[:, crop_x: label.shape[1] - crop_x, crop_y: label.shape[2] - crop_y]
    
     
        #label3.show()

       



        
        label=label.squeeze()
   

        #pred size is 2 2 572 572
        # label is 2 572 572

        loss = criterion(pred, label)

        loss.backward()



        optimizer.step()
        optimizer.zero_grad()

        epoch_loss += loss.item()




        





        print('batch %d --- Loss: %.4f' % (i, loss.item() / batch_size))
    print('Epoch %d / %d --- Loss: %.4f' % (e + 1, epoch_n, epoch_loss / trainset.__len__()))

    train_loss.append(epoch_loss / trainset.__len__())


    torch.save(model.state_dict(), 'checkpoint.pt')

    model.eval()

    total = 0
    correct = 0
    total_loss = 0

    with torch.no_grad():
        for i, data in enumerate(testloader):
            image, label = data

            #image = image.unsqueeze(1).to(device)
            image=image.to(device)       
            label = label.long().to(device)

            
            pred = model(image)

            crop_x = (label.shape[1] - pred.shape[2]) // 2
            crop_y = (label.shape[2] - pred.shape[3]) // 2

            label = label[:, crop_x: label.shape[1] - crop_x, crop_y: label.shape[2] - crop_y]

            label=label.squeeze()
            loss = criterion(pred, label)
            total_loss += loss.item()




            _, pred_labels = torch.max(pred, dim=1)

            total += label.shape[0] * label.shape[1] * label.shape[2]
            correct += (pred_labels == label).sum().item()

        print('Accuracy: %.4f ---- Loss: %.4f' % (correct / total, total_loss / testset.__len__()))

        test_loss.append(total_loss / testset.__len__())


train_loss_numpy=np.array(train_loss)
test_loss_numpy=np.array(test_loss)



#testing and visualization

model.eval()

output_masks = []
output_labels = []




with torch.no_grad():
    for i in range(testset.__len__()):
        image, labels = testset.__getitem__(i)

        input_image = image.unsqueeze(0).to(device)
        pred = model(input_image)

        output_mask = torch.max(pred, dim=1)[1].cpu().squeeze(0).numpy()


        crop_x = (labels.shape[0] - output_mask.shape[0]) // 2
        crop_y = (labels.shape[1] - output_mask.shape[1]) // 2
        labels = labels[crop_x: labels.shape[0] - crop_x, crop_y: labels.shape[1] - crop_y].cpu().numpy()

        output_masks.append(output_mask)
        output_labels.append(labels)


    for i in range(trainset.__len__()):
        image, labels = trainset.__getitem__(i)

        input_image = image.unsqueeze(0).to(device)
        pred = model(input_image)

        output_mask = torch.max(pred, dim=1)[1].cpu().squeeze(0).numpy()


        crop_x = (labels.shape[0] - output_mask.shape[0]) // 2
        crop_y = (labels.shape[1] - output_mask.shape[1]) // 2
        labels = labels[crop_x: labels.shape[0] - crop_x, crop_y: labels.shape[1] - crop_y].cpu().numpy()

        output_masks.append(output_mask)
        output_labels.append(labels)


#fig, axes = plt.subplots(testset.__len__(), 2, figsize = (20, 20))



# transform1=torchvision.transforms.ToPILImage()
# for i in range(testset.__len__()):
#     output_labels[i]=torch.from_numpy(output_labels[i])
#     output_labels[i]=output_labels[i].squeeze()



# for i in range(testset.__len__()):
#     output_labels[i]=transform1(output_labels[i])
#     #output_labels[i].show()




# for i in range(testset.__len__()):
#     output_masks[i]=torch.from_numpy(output_masks[i])
#     output_masks[i]=output_masks[i].squeeze()
#     output_masks[i]=output_masks[i].squeeze()


# for i in range(testset.__len__()):
#     output_masks[i]=output_masks[i].float()
    
#     output_masks[i]=transform1(output_masks[i])
#     output_masks[i]=ImageEnhance.Brightness(output_masks[i]).enhance(1000)
#     #output_masks[i].show()




# gt=output_labels[0].cpu().numpy()
      
# plt.imshow(gt,cmap='gray')
# plt.show()

# print(output_masks[0].size())
# pred1=output_masks[0][0]
# pred=pred1.cpu().numpy()
      
# plt.imshow(pred,cmap='gray')
# plt.show()
# print("11111111111111111111")


"""
for i in range(testset.__len__()):


  axes[i, 0].imshow(output_labels[i][0],cmap='gray')
  axes[i, 0].axis('off')
 

  axes[i, 1].imshow(output_masks[i],cmap='gray')
  
  axes[i, 1].axis('off')

plt.show()
"""

for i in range(testset.__len__()+trainset.__len__()):
  plt.imshow(output_labels[i][0],cmap='gray')
  plt.show()

  plt.imshow(output_masks[i],cmap='gray')
  plt.show()



plt.plot(train_loss_numpy)
plt.plot(test_loss_numpy)
plt.show()




"""for i in range(testset.__len__()):
  
    output_labels[i]=np.squeeze(output_labels[i],axis=0)










a=torch.from_numpy(output_labels[0])
transform1=torchvision.transforms.ToPILImage()
b=transform1(a)
b.show()


c=torch.from_numpy(output_masks[0])
c=c.float()
d=transform1(c)
d.show()





"""