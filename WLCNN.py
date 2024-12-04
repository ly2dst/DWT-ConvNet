
import numpy
import torch
import pytorch_wavelets as pw
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self,length=4,size=256):
        super(Model, self).__init__()
        self.size=size
        self.dwt=pw.DWT2D(1,'haar','zero')
        self.idw=pw.IDWT2D('haar','zero')
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1)
        self.conv2=nn.Conv2d(4,16,kernel_size=3,stride=1,padding=1)

        self.conv3=nn.Conv2d(16,64,kernel_size=3,stride=1,padding=1)
        self.conv4=nn.Conv2d(64,256,kernel_size=3,stride=1,padding=1)

        self.conv_LL1=nn.Conv2d(1,4,kernel_size=3,stride=1,padding=1)
        self.conv_LH1=nn.Conv2d(1,4,kernel_size=3,stride=1,padding=1)
        self.conv_HL1=nn.Conv2d(1,4,kernel_size=3,stride=1,padding=1)
        self.conv_HH1=nn.Conv2d(1,4,kernel_size=3,stride=1,padding=1)

        self.conv_LL2=nn.Conv2d(1,4,kernel_size=3,stride=1,padding=1)
        self.conv_LH2=nn.Conv2d(1,4,kernel_size=3,stride=1,padding=1)
        self.conv_HL2=nn.Conv2d(1,4,kernel_size=3,stride=1,padding=1)
        self.conv_HH2=nn.Conv2d(1,4,kernel_size=3,stride=1,padding=1)

        self.conv_LL11=nn.Conv2d(4,16,kernel_size=3,stride=1,padding=1)
        self.conv_LH11=nn.Conv2d(4,16,kernel_size=3,stride=1,padding=1)
        self.conv_HL11=nn.Conv2d(4,16,kernel_size=3,stride=1,padding=1)
        self.conv_HH11=nn.Conv2d(4,16,kernel_size=3,stride=1,padding=1)

        self.conv_LL22=nn.Conv2d(4,16,kernel_size=3,stride=1,padding=1)
        self.conv_LH22=nn.Conv2d(4,16,kernel_size=3,stride=1,padding=1)
        self.conv_HL22=nn.Conv2d(4,16,kernel_size=3,stride=1,padding=1)
        self.conv_HH22=nn.Conv2d(4,16,kernel_size=3,stride=1,padding=1)
        
        self.dropout1d=nn.Dropout(0.5)
        self.dropout2d_1=nn.Dropout2d(0.5)
        self.dropout2d_2=nn.Dropout2d(0.3)
        self.fc1 = nn.Linear(256*128*128,256)
        self.fc2=nn.Linear(256,length)
        self.fc3 = nn.Linear(256, length)  

    # def getWL(self,img):
    #     dwt=pw.DWT2D(1,'haar','zero')
    #     coeff1=dwt(img)
    #     L1=coeff1[0]
    #     coeff2=dwt(L1)
    #     L2=coeff2[0]
    #     return (L1,coeff1[1][0],L2,coeff2[1][0])
    
    def getIDW1(self,img,layer):
        Hlist=[]
        if layer==1:
            ll,coeff=self.dwt(img)
            H=coeff[0]
            ll2=self.getIDW1(ll,2)
            img_LH=self.conv_LH1(H[:,:,0,:,:])#output size:[batch_size,4,size/2^n,size/2^n]
            img_HL=self.conv_HL1(H[:,:,1,:,:])#output size:[batch_size,4,size/2^n,size/2^n]
            img_HH=self.conv_HH1(H[:,:,2,:,:])#output size:[batch_size,4,size/2^n,size/2^n]
            img_LL=self.conv_LL1(ll)#size:[batch_size,1,size/2,size/2]
            for i in range(0,img_LH.shape[1]):
                temp=torch.cat((img_LH[:,i,:,:].unsqueeze(1).unsqueeze(1),img_HL[:,i,:,:].unsqueeze(1).unsqueeze(1),
                                img_HH[:,i,:,:].unsqueeze(1).unsqueeze(1)),2)
                idw=self.idw((ll2[:,i,:,:].unsqueeze(1)+img_LL[:,i,:,:].unsqueeze(1),[temp]))
                Hlist.append(idw)
            merged_img=torch.cat([tensor for tensor in Hlist],1)
            return merged_img
        
        elif layer==2:
            ll,coeff=self.dwt(img)
            H=coeff[0]
            img_LH=self.conv_LH2(H[:,:,0,:,:]).unsqueeze(1)#output size:[batch_size,1,3,size/2^n,size/2^n]
            img_HL=self.conv_HL2(H[:,:,1,:,:]).unsqueeze(1)#output size:[batch_size,1,3,size/2^n,size/2^n]
            img_HH=self.conv_HH2(H[:,:,2,:,:]).unsqueeze(1)#output size:[batch_size,1,3,size/2^n,size/2^n]
            img_LL=self.conv_LL2(ll)#size:[batch_size,1,size/2,size/2]
            for i in range(0,img_LH.shape[2]):
                temp=torch.cat((img_LH[:,:,i,:,:].unsqueeze(1),img_HL[:,:,i,:,:].unsqueeze(1),
                                img_HH[:,:,i,:,:].unsqueeze(1)),2)
                idw=self.idw((img_LL[:,i,:,:].unsqueeze(1),[temp]))
                Hlist.append(idw)
            merged_img=torch.cat([tensor for tensor in Hlist],1)
            return merged_img
        

    def getIDW2(self,img,layer):
        
        Hlist=[]
        # print(f'img.shape: {img.shape}')
        if layer==1:
            ll,coeff=self.dwt(img)
            # print(f'll.shape: {ll.shape}')
            H=coeff[0]
            ll2=self.getIDW2(ll,2)
            # print(f'll2.shape:{ll2.shape}')
            
            img_LH=self.conv_LH11(H[:,:,0,:,:])#output size:[batch_size,4,size/2^n,size/2^n]
            img_HL=self.conv_HL11(H[:,:,1,:,:])#output size:[batch_size,4,size/2^n,size/2^n]
            img_HH=self.conv_HH11(H[:,:,2,:,:])#output size:[batch_size,4,size/2^n,size/2^n]
            img_LL=self.conv_LL11(ll)#size:[batch_size,1,size/2,size/2]

            for i in range(0,img_LH.shape[1]):
                temp=torch.cat((img_LH[:,i,:,:].unsqueeze(1).unsqueeze(1),img_HL[:,i,:,:].unsqueeze(1).unsqueeze(1),
                                img_HH[:,i,:,:].unsqueeze(1).unsqueeze(1)),2)
                idw=self.idw((ll2[:,i,:,:].unsqueeze(1)+img_LL[:,i,:,:].unsqueeze(1),[temp]))
                Hlist.append(idw)
            merged_img=torch.cat([tensor for tensor in Hlist],1)
            return merged_img
        
        elif layer==2:
            ll,coeff=self.dwt(img)
            H=coeff[0]
            img_LH=self.conv_LH22(H[:,:,0,:,:]).unsqueeze(1)#output size:[batch_size,1,3,size/2^n,size/2^n]
            img_HL=self.conv_HL22(H[:,:,1,:,:]).unsqueeze(1)#output size:[batch_size,1,3,size/2^n,size/2^n]
            img_HH=self.conv_HH22(H[:,:,2,:,:]).unsqueeze(1)#output size:[batch_size,1,3,size/2^n,size/2^n]
            img_LL=self.conv_LL22(ll)#size:[batch_size,1,size/2,size/2]
            for i in range(0,img_LH.shape[2]):
                temp=torch.cat((img_LH[:,:,i,:,:].unsqueeze(1),img_HL[:,:,i,:,:].unsqueeze(1),
                                img_HH[:,:,i,:,:].unsqueeze(1)),2)
                idw=self.idw((img_LL[:,i,:,:].unsqueeze(1),[temp]))
                Hlist.append(idw)
            merged_img=torch.cat([tensor for tensor in Hlist],1)
            return merged_img
        
        
    def forward(self, x):
        merged_img=F.relu(self.conv1(x))
        idw1=F.relu(self.getIDW1(x,layer=1))
        merged_img=idw1+merged_img
        self.dropout2d_2(merged_img)
        # merged_img=torch.max_pool2d(merged_img,2)

        # idw2=F.relu(self.getIDW2(merged_img,layer=1))
        merged_img=F.relu(self.conv2(merged_img)) 
        # merged_img=merged_img+idw2
        merged_img=self.conv3(merged_img)
        merged_img=torch.max_pool2d(merged_img,2)
        merged_img=self.conv4(merged_img)
        merged_img=torch.max_pool2d(merged_img,2)
        # merged_img=torch.max_pool2d(merged_img,2)
        merged_img=self.dropout2d_1(merged_img)

        merged_img=merged_img.view(merged_img.size(0),-1)

        y=self.fc1(merged_img)
        y=self.dropout1d(y)
        y=self.fc2(y)
        return y
