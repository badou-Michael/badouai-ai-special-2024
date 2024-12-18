import torch
import torch.nn as nn


def BasicConv2d(x,input_channels,output_channels,**kwargs)
    x=nn.Conv2d(input_channels,output_channels,bias=False,**kwargs)(x)
    x=nn.BatchNorm2d(output_channels)(x)
    x=nn.ReLU(inplace=True)(x)
    return x

def InceptionA(x,input_channels,output_channels):
    branch1x1=BasicConv2d(x,input_channels,64,kernel_size=1)
    
    branch5x5=BasicConv2d(x,input_channels,48,kernel_size=1)
    branch5x5=BasicConv2d(branch5x5,48,64,kernel_size=5,padding=2)
    
    branch3x3=BasicConv2d(x,input_channels,64,kernel_size=1)
    branch3x3=BasicConv2d(branch3x3,64,96,kernel_size=3,padding=1)
    branch3x3=BasicConv2d(branch3x3,96,96,kernel_size=3,padding=1)
    
    branch_pool=nn.AvgPool2d(kernel_size=3,stride=1,padding=1)(x)
    branch_pool=BasicConv2d(branch_pool,input_channels,output_channels,kernel_size=3,padding=1)
    
    output=torch.cat([branch1x1,branch5x5,branch3x3,branch_pool],dim=1)
    return output


def InceptionB(x,input_channels):
    branch3x3=BasicConv2d(x,input_channels,384,kernel_size=3,stride=2)
    
    branch3x3stack=BasicConv2d(x,input_channels,64,kernel_size=1)
    branch3x3stack=BasicConv2d(branch3x3stack,64,96,kernel_size=3,padding=1)
    branch3x3stack=BasicConv2d(branch3x3stack,96,96,kernel_size=3,stride=2)
    
    branch_pool=nn.MaxPool2d(kernel_size=3,stride=2)(x)
    
    output=torch.cat([branch3x3,branch3x3stack,branch_pool],dim=1)
    return output


def InceptionC(x,input_channels,output_channels):
    branch1x1=BasicConv2d(x,input_channels,192,kernel_size=1)
    
    branch7x7=BasicConv2d(x,input_channels,output_channels,kernel_size=1)
    branch7x7=BasicConv2d(branch7x7,output_channels,output_channels,kernel_size=(1,7),padding=(0,3))
    branch7x7=BasicConv2d(branch7x7,output_channels,192,kernel_size=(1,7),padding=(0,3))
    
    branch7x7stack=BasicConv2d(x,input_channels,output_channels,kernel_size=1)
    branch7x7stack=BasicConv2d(branch7x7stack,output_channels,output_channels,kernel_size=(7,1),padding=(3,0))
    branch7x7stack=BasicConv2d(branch7x7stack,output_channels,output_channels,kernel_size=(1,7),padding=(0,3))
    branch7x7stack=BasicConv2d(branch7x7stack,output_channels,output_channels,kernel_size=(7,1),padding=(3,0))
    branch7x7stack=BasicConv2d(branch7x7stack,output_channels,192,kernel_size=(1,7),padding=(0,3))
    
    branch_pool=nn.AvgPool2d(kernel_size=3,stride=1,padding=1)(x)
    branch_pool=BasicConv2d(branch_pool,input_channels,192,kernel_size=1)
    
    output=torch.cat([branch1x1,branch7x7,branch7x7stack,branch_pool],dim=1)
    return output


def InceptionD(x,input_channels):
    branch3x3=BasicConv2d(x,input_channels,192,kernel_size=1)
    branch3x3=BasicConv2d(branch3x3,192,320,kernel_size=3,stride=2)
    
    branch7x7=BasicConv2d(x,input_channels,192,kernel_size=1)
    branch7x7=BasicConv2d(branch7x7,192,192,kernel_size=(1,7),padding=(0,3))
    branch7x7=BasicConv2d(branch7x7,192,192,kernel_size=(7,1),padding=(3,0))
    branch7x7=BasicConv2d(branch7x7,192,192,kernel_size=3,stride=2)
    
    branch_pool=nn.AvgPool2d(kernel_size=3,stride=2)(x)
    
    output=torch.cat([branch3x3,branch7x7,branch_pool],dim=1)
    return output


def InceptionE(x,input_channels):
    branch1x1=BasicConv2d(x,input_channels,320,kernel_size=1)
    
    branch3x3_1=BasicConv2d(x,input_channels,384,kernel_size=1)
    branch3x3_2a=BasicConv2d(branch3x3_1,384,384,kernel_size=(1,3),padding=(0,1))
    branch3x3_2b=BasicConv2d(branch3x3_1,384,384,kernel_size=(3,1),padding=(1,0))
    branch3x3=torch.cat([branch3x3_2a,branch3x3_2b],dim=1)
    
    branch3x3stack_1=BasicConv2d(x,input_channels,448,kernel_size=1)
    branch3x3stack_2=BasicConv2d(branch3x3stack_1,448,384,kernel_size=3,padding=1)
    branch3x3stack_3a=BasicConv2d(branch3x3stack_2,384,384,kernel_size=(1,3),padding=(0,1))
    branch3x3stack_3b=BasicConv2d(branch3x3stack_2,384,384,kernel_size=(3,1),padding=(1,0))
    branch3x3stack=torch.cat([branch3x3stack_3a,branch3x3stack_3b],dim=1)
    
    branch_pool=nn.AvgPool2d(kernel_size=3,stride=1,padding=1)(x)
    branch_pool=BasicConv2d(branch_pool,input_channels,192,kernel_size=1)
    
    output=torch.cat([branch1x1,branch3x3,branch3x3stack,branch_pool],dim=1)
    return output


class InceptionV3(nn.Module):
    
    def __init__(self,num_classes=1000):
        super().__init__()
        self.Conv2d_1a_3x3=BasicConv2d(3,32,kernel_size=3,padding=1)
        self.Conv2d_2a_3x3=BasicConv2d(32,32,kernel_size=3,padding=1)
        self.Conv2d_2b_3x3=BasicConv2d(32,64,kernel_size=3,padding=1)
        self.Conv2d_3b_1x1=BasicConv2d(64,80,kernel_size=1)
        self.Conv2d_4a_3x3=BasicConv2d(80,192,kernel_size=3)
        
        self.Mixed_5b=InceptionA(192,output_channels=32)
        self.Mixed_5c=InceptionA(256,output_channels=64)
        self.Mixed_5d=InceptionA(288,output_channels=64)
        
        self.Mixed_6a=InceptionB(288)
        
        self.Mixed_6b=InceptionC(768,output_channels=128)
        self.Mixed_6c=InceptionC(768,output_channels=160)
        self.Mixed_6d=InceptionC(768,output_channels=160)
        self.Mixed_6e=InceptionC(768,output_channels=192)
        
        self.Mixed_7a=InceptionD(768)
        self.Mixed_7b=InceptionE(1280)
        self.Mixed_7c=InceptionE(2048)
        
        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.dropout=nn.Dropout()
        self.fc=nn.Linear(2048,num_classes)
        
    def foward(self,x):
        x=self.Conv2d_1a_3x3(x)
        x=self.Conv2d_2a_3x3(x)
        x=self.Conv2d_2b_3x3(x)
        x=self.Conv2d_3b_1x1(x)
        x=self.Conv2d_4a_3x3(x)
        
        x=self.Mixed_5b(x)
        x=self.Mixed_5c(x)
        x=self.Mixed_5d(x)
        
        x=self.Mixed_6a(x)
        
        x=self.Mixed_6b(x)
        x=self.Mixed_6c(x)
        x=self.Mixed_6d(x)
        x=self.Mixed_6e(x)
        
        x=self.Mixed_7a(x)
        
        x=self.Mixed_7b(x)
        x=self.Mixed_7c(x)
        
        x=self.avgpool(x)
        x=self.dropout(x)
        x=torch.flatten(x,1)
        x=self.fc(x)
        return x
    