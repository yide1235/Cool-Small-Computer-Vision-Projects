import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np

import os
import random

from PIL import Image, ImageOps

#import any other libraries you need below this line
from PIL import ImageEnhance
import math
import torchvision
from torch import nn 
from torchvision import transforms as T






class Cell_data(Dataset):
    def __init__(self, data_dir, size, train='True', train_test_split=0.8, augment_data=True):
        ##########################inputs##################################
        # data_dir(string) - directory of the data#########################
        # size(int) - size of the images you want to use###################
        # train(boolean) - train data or test data#########################
        # train_test_split(float) - the portion of the data for training###
        # augment_data(boolean) - use data augmentation or not#############
        super(Cell_data, self).__init__()

        self.augment_data=augment_data

        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            #i use resize here not crop to make sure all features are obtained
            torchvision.transforms.Resize(size),
            #torchvision.transforms.Normalize((0.5,),(0.5,))

        ])

        #just for resize
        transform2 = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            #i use resize here not crop to make sure all features are obtained
            torchvision.transforms.Resize(size),
            #torchvision.transforms.Normalize((0.5,),(0.5,))

        ])
        

     

        n=len(os.listdir(data_dir+"/scans"))
        self.images=torch.tensor([]).cuda()
        self.masks=torch.tensor([]).cuda()

        

        #set a string
        dir=os.listdir(data_dir+"/scans")
        dir2=os.listdir(data_dir+"/labels")

        for i in range(n):
            
            

            img1=Image.open(data_dir+"/scans/"+dir[i])

         
            img=transform(img1)

        
            #now img size is 1,512,512
        

            
            
            
            


            mask1=Image.open(data_dir+"/labels/"+dir2[i])
            
        
          
          
           
            #different for image because we dont do normalize for image
            mask2=ImageEnhance.Brightness(mask1).enhance(1000)
       
            mask3=transform2(mask2)
            mask4=torchvision.transforms.ToPILImage()(mask3)
            #mask4.show()
    
            mask=torchvision.transforms.ToTensor()(mask4)
      

            img=img.cuda()
            mask=mask.cuda()
        
          
        

            self.images=torch.cat((self.images,img),0)
            self.masks=torch.cat((self.masks,mask),0)
   
       
        #the size of self.images and self.mask is 38,512,512

            
      
    
  
        split=int(n*train_test_split)
        #now split is at 80%
     

        if train:
            
            self.images=self.images[:split]
            self.masks=self.masks[:split]
          
        else:
            self.images=self.images[split:]
            self.masks=self.masks[split:]


        # todo

        
        # initialize the data class

    def __len__(self):
        return len(self.images)


    def __getitem__(self, idx):
        # todoSS


        # load image and mask from index idx of your data
        image=self.images[idx]
        mask=self.masks[idx]


  
        
        image,mask=image.cuda(),mask.cuda()

        image=image.unsqueeze(0)
        mask=mask.unsqueeze(0)


        # data augmentation part
        if self.augment_data:
            augment_mode = np.random.randint(0, 4)
            if augment_mode == 0:
                # todo
                image=torchvision.transforms.functional.vflip(image)
                mask=torchvision.transforms.functional.vflip(mask)
                # flip image vertically
            elif augment_mode == 1:
                # todo
                image=torchvision.transforms.functional.adjust_gamma(image,1.5)
                mask=torchvision.transforms.functional.adjust_gamma(mask,1.5)
                # flip image horizontally
            elif augment_mode == 2:
                # todo
                image=torchvision.transforms.functional.rotate(image,270)
                mask=torchvision.transforms.functional.rotate(mask,270)
                # zoom image
            else:
                # todo
                image=torchvision.transforms.functional.rotate(image,90)
                mask=torchvision.transforms.functional.rotate(mask,90)
                # rotate image

        #image and mask are 1 572 572



        # myd=torchvision.transforms.ToPILImage()(image)
        # myd2=torchvision.transforms.ToPILImage()(mask)
        # myd.show()
        # myd2.show()
        #here image is not wrong yet


        # todo
        # return image and mask in tensors
        return image,mask
       









