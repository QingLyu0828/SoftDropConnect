import os
import random
import numpy as np
import torch
import h5py
from torch.utils.data import Dataset
from torchvision import transforms
import xlrd
import nibabel as nib
from skimage import transform
from .cropping import crop

# def random_rot(img1,img2,img3,img4,img5):
#     k = np.random.randint(0, 4)
#     img1 = np.rot90(img1, k)
#     img2 = np.rot90(img2, k)
#     img3 = np.rot90(img3, k)
#     img4 = np.rot90(img4, k)
#     img5 = np.rot90(img5, k)
#     return img1,img2,img3,img4,img5

# def random_flip(img1,img2,img3,img4,img5):
#     axis = np.random.randint(0, 2)
#     img1 = np.flip(img1, axis=axis).copy()
#     img2 = np.flip(img2, axis=axis).copy()
#     img3 = np.flip(img3, axis=axis).copy()
#     img4 = np.flip(img4, axis=axis).copy()
#     img5 = np.flip(img5, axis=axis).copy()
#     return img1,img2,img3,img4,img5

# class RandomGenerator(object):
#     def __init__(self, output_size):
#         self.output_size = output_size

#     def __call__(self, sample):
#         img1, img2, img3 = sample['img1'], sample['img2'], sample['img3']
#         img4, img5 = sample['img4'], sample['img5']

#         if random.random() > 0.5:
#             img1,img2,img3,img4,img5 = random_rot(img1,img2,img3,img4,img5)
#         if random.random() > 0.5:
#             img1,img2,img3,img4,img5 = random_flip(img1,img2,img3,img4,img5)
#         sample = {'img1': img1,'img2': img2,'img3': img3,'img4': img4,'img5': img5}
#         return sample



class Train_Data(Dataset):
    def __init__(self):       

        # self.transform=transforms.Compose([RandomGenerator(output_size=[240, 240])])
        self.len = 1200
        
    def __getitem__(self, index):

        path = '/gpfs/u/scratch/DTIR/DTIRqngl/Data/BraTS2021/TrainingData'
        # path = 'G:/Data/BraTS2021/RSNA_ASNR_MICCAI_BraTS2021_TrainingData'
        folders = os.listdir(path)
        
        name = path + '/' + folders[index] + '/' + folders[index] + '_t1.nii.gz'
        data1 = nib.load(name).get_fdata()
        img1 = data1
       
        name = path + '/' + folders[index] + '/' + folders[index] + '_t1ce.nii.gz'
        data2 = nib.load(name).get_fdata()
        img2 = data2
        
        name = path + '/' + folders[index] + '/' + folders[index] + '_t2.nii.gz'
        data3 = nib.load(name).get_fdata()
        img3 = data3
        
        name = path + '/' + folders[index] + '/' + folders[index] + '_flair.nii.gz'
        data4 = nib.load(name).get_fdata()
        img4 = data4
        
        name = path + '/' + folders[index] + '/' + folders[index] + '_seg.nii.gz'
        data5 = nib.load(name).get_fdata()
        img5 = data5
        img5[img5==4]=3
         
        data = np.zeros((4,240,240,155))
        data[0,:,:] = img1.copy()
        data[1,:,:] = img2.copy()
        data[2,:,:] = img3.copy()
        data[3,:,:] = img4.copy()
        label = np.zeros((240,240,155))
        label = img5.copy()
        
        data1, label1 = crop(data, label)
        nx, ny, nz = label1.shape
        
        if nx <= 128:
            ndata = np.zeros((4,128,128,nz))
            nlabel = np.zeros((128,128,nz))
            resy = np.random.randint(ny-128)
            ndata[:,0:nx,:,:] = data1[:,:,resy:resy+128,:]
            nlabel[0:nx,:,:] = label1[:,resy:resy+128,:]
        else:
            ndata = np.zeros((4,128,128,nz))
            nlabel = np.zeros((128,128,nz))
            resx = np.random.randint(nx-128)
            resy = np.random.randint(ny-128)
            ndata = data1[:,resx:resx+128,resy:resy+128,:]
            nlabel = label1[resx:resx+128,resy:resy+128,:]
        if nz <= 128:
            nndata = np.zeros((4,128,128,128))
            nnlabel = np.zeros((128,128,128))
            nndata[:,:,:,0:nz] = ndata
            nnlabel[:,:,0:nz] = nlabel
        else:
            nndata = np.zeros((4,128,128,128))
            nnlabel = np.zeros((128,128,128))
            resz = np.random.randint(nz-128)
            nndata = ndata[:,:,:,resz:resz+128]
            nnlabel = nlabel[:,:,resz:resz+128]            
        
        # sample = {'data': nndata,'label': nnlabel}
        # if self.transform:
        #     sample = self.transform(sample)  
        # img1, img2, img3 = sample['img1'], sample['img2'], sample['img3']
        # img4, img5 = sample['img4'], sample['img5']
        
        nndata = np.transpose(nndata, (0,3,1,2))
        nnlabel = np.transpose(nnlabel, (2,0,1))
        for i in range(4):
            nndata[i] = self.norm(nndata[i])
            
        x = torch.from_numpy(nndata)
        y = torch.from_numpy(nnlabel)
        
        x = x.type(torch.FloatTensor)
        y = y.type(torch.LongTensor)
        
        return x, y
    
    def __len__(self):
        return self.len
    
    def norm(self, x):
        if np.amax(x) > 0:
            x = (x - np.amin(x)) / (np.amax(x) - np.amin(x))
        return x
    
    
class Test_Data(Dataset):
    def __init__(self):       
        self.len = 5
        
    def __getitem__(self, idx):
        index = idx + 1200
        
        path = '/gpfs/u/scratch/DTIR/DTIRqngl/Data/BraTS2021/TrainingData'
        # path = 'G:/Data/BraTS2021/RSNA_ASNR_MICCAI_BraTS2021_TrainingData'
        folders = os.listdir(path)
        
        name = path + '/' + folders[index] + '/' + folders[index] + '_t1.nii.gz'
        data1 = nib.load(name).get_fdata()
        img1 = data1
        # img1 = self.norm(img1)
        
        name = path + '/' + folders[index] + '/' + folders[index] + '_t1ce.nii.gz'
        data2 = nib.load(name).get_fdata()
        img2 = data2
        # img2 = self.norm(img2)
        
        name = path + '/' + folders[index] + '/' + folders[index] + '_t2.nii.gz'
        data3 = nib.load(name).get_fdata()
        img3 = data3
        # img3 = self.norm(img3)
        
        name = path + '/' + folders[index] + '/' + folders[index] + '_flair.nii.gz'
        data4 = nib.load(name).get_fdata()
        img4 = data4
        # img4 = self.norm(img4)
        
        name = path + '/' + folders[index] + '/' + folders[index] + '_seg.nii.gz'
        data5 = nib.load(name).get_fdata()
        img5 = data5
        img5[img5==4]=3
         
        data = np.zeros((4,240,240,155))
        data[0,:,:] = img1.copy()
        data[1,:,:] = img2.copy()
        data[2,:,:] = img3.copy()
        data[3,:,:] = img4.copy()
        label = np.zeros((240,240,155))
        label = img5.copy()

        data1, label1 = crop(data, label)
        nx, ny, nz = label1.shape
        
        if nx <= 128:
            ndata = np.zeros((4,128,128,nz))
            nlabel = np.zeros((128,128,nz))
            resy = np.random.randint(ny-128)
            ndata[:,0:nx,:,:] = data1[:,:,resy:resy+128,:]
            nlabel[0:nx,:,:] = label1[:,resy:resy+128,:]
        else:
            ndata = np.zeros((4,128,128,nz))
            nlabel = np.zeros((128,128,nz))
            resx = (nx-128)//2
            resy = (ny-128)//2
            ndata = data1[:,resx:resx+128,resy:resy+128,:]
            nlabel = label1[resx:resx+128,resy:resy+128,:]
        if nz <= 128:
            nndata = np.zeros((4,128,128,128))
            nnlabel = np.zeros((128,128,128))
            nndata[:,:,:,0:nz] = ndata
            nnlabel[:,:,0:nz] = nlabel
        else:
            nndata = np.zeros((4,128,128,128))
            nnlabel = np.zeros((128,128,128))
            resz = (nz-128)//2
            nndata = ndata[:,:,:,resz:resz+128]
            nnlabel = nlabel[:,:,resz:resz+128]
            
        nndata = np.transpose(nndata, (0,3,1,2))
        nnlabel = np.transpose(nnlabel, (2,0,1))
        for i in range(4):
            nndata[i] = self.norm(nndata[i])
            
        x = torch.from_numpy(nndata)
        y = torch.from_numpy(nnlabel)
        
        x = x.type(torch.FloatTensor)
        y = y.type(torch.LongTensor)
        
        return x, y
    
    def __len__(self):
        return self.len
    
    def norm(self, x):
        if np.amax(x) > 0:
            x = (x - np.amin(x)) / (np.amax(x) - np.amin(x))
        return x


class Valid_Data(Dataset):
    def __init__(self):       
        self.len = 50
        
    def __getitem__(self, idx):
        index = idx + 1200

        path = '/gpfs/u/scratch/DTIR/DTIRqngl/Data/BraTS2021/TrainingData'
        # path = 'G:/Data/BraTS2021/RSNA_ASNR_MICCAI_BraTS2021_TrainingData'
        folders = os.listdir(path)
        
        name = path + '/' + folders[index] + '/' + folders[index] + '_t1.nii.gz'
        data1 = nib.load(name).get_fdata()
        img1 = data1
        
        name = path + '/' + folders[index] + '/' + folders[index] + '_t1ce.nii.gz'
        data2 = nib.load(name).get_fdata()
        img2 = data2
        
        name = path + '/' + folders[index] + '/' + folders[index] + '_t2.nii.gz'
        data3 = nib.load(name).get_fdata()
        img3 = data3
        
        name = path + '/' + folders[index] + '/' + folders[index] + '_flair.nii.gz'
        data4 = nib.load(name).get_fdata()
        img4 = data4
        
        name = path + '/' + folders[index] + '/' + folders[index] + '_seg.nii.gz'
        data5 = nib.load(name).get_fdata()
        img5 = data5
        img5[img5==4]=3
         
        data = np.zeros((4,240,240,155))
        data[0,:,:] = img1.copy()
        data[1,:,:] = img2.copy()
        data[2,:,:] = img3.copy()
        data[3,:,:] = img4.copy()
        label = np.zeros((240,240,155))
        label = img5.copy()

        data1, label1 = crop(data, label)
        nx, ny, nz = label1.shape
        
        if nx <= 128:
            ndata = np.zeros((4,128,128,nz))
            nlabel = np.zeros((128,128,nz))
            resy = np.random.randint(ny-128)
            ndata[:,0:nx,:,:] = data1[:,:,resy:resy+128,:]
            nlabel[0:nx,:,:] = label1[:,resy:resy+128,:]
        else:
            ndata = np.zeros((4,128,128,nz))
            nlabel = np.zeros((128,128,nz))
            resx = np.random.randint(nx-128)
            resy = np.random.randint(ny-128)
            ndata = data1[:,resx:resx+128,resy:resy+128,:]
            nlabel = label1[resx:resx+128,resy:resy+128,:]
        if nz <= 128:
            nndata = np.zeros((4,128,128,128))
            nnlabel = np.zeros((128,128,128))
            nndata[:,:,:,0:nz] = ndata
            nnlabel[:,:,0:nz] = nlabel
        else:
            nndata = np.zeros((4,128,128,128))
            nnlabel = np.zeros((128,128,128))
            resz = np.random.randint(nz-128)
            nndata = ndata[:,:,:,resz:resz+128]
            nnlabel = nlabel[:,:,resz:resz+128]
            
        nndata = np.transpose(nndata, (0,3,1,2))
        nnlabel = np.transpose(nnlabel, (2,0,1))
        for i in range(4):
            nndata[i] = self.norm(nndata[i])
            
        x = torch.from_numpy(nndata)
        y = torch.from_numpy(nnlabel)
        
        x = x.type(torch.FloatTensor)
        y = y.type(torch.LongTensor)
        
        return x, y
    
    def __len__(self):
        return self.len
    
    def norm(self, x):
        if np.amax(x) > 0:
            x = (x - np.amin(x)) / (np.amax(x) - np.amin(x))
        return x
