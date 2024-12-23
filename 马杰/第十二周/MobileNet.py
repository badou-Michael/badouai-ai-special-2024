import torch
import torch.nn as nn

def DepthSeperableConv2d(x,input_channels,output_channels,kernel_size,**kargs):
    depthwise=nn.ReLU(inplace=True)(
            nn.BatchNorm2d(input_channels)(
                nn.Conv2d(input_channels,input_channels,kernel_size,groups=input_channels,**kargs)(x)
            )
        )
    
    return nn.ReLU(inplace=True)(
            nn.BatchNorm2d(output_channels)(
                nn.Conv2d(input_channels,output_channels,1)(depthwise)
            )
        )


def BasicConv2d(x,input_channels,output_channels,kernel_size,**kwargs):
    return nn.ReLU(inplace=True)(
        nn.BatchNorm2d(output_channels)(
            nn.Conv2d(input_channels,output_channels,kernel_size,**kwargs)(x)
        )
    )
    
class MobileNet(nn.Module):
    def __init__(self,width_multiplier=1,num_classes=1000):
        super(MobileNet,self).__init__()
        
        alpha=width_multiplier
        self.stem=nn.Sequential(
            BasicConv2d(3,int(32*alpha),3,padding=1,bias=False),
            DepthSeperableConv2d(int(32*alpha),int(64*alpha),3,padding=1,bias=False),
        )
        
        self.conv1=nn.Sequential(
            DepthSeperableConv2d(int(64*alpha),int(128*alpha),3,stride=2,padding=1,bias=False),
            DepthSeperableConv2d(int(128*alpha),int(128*alpha),3,padding=1,bias=False),
        )
        
        self.conv2=nn.Sequential(
            DepthSeperableConv2d(int(128*alpha),int(256*alpha),3,stride=2,padding=1,bias=False),
            DepthSeperableConv2d(int(256*alpha),int(256*alpha),3,padding=1,bias=False),
        )
        
        self.conv3=nn.Sequential(
            DepthSeperableConv2d(int(256*alpha),int(512*alpha),3,stride=2,padding=1,bias=False),
            DepthSeperableConv2d(int(512*alpha),int(512*alpha),3,padding=1,bias=False),
            DepthSeperableConv2d(int(512*alpha),int(512*alpha),3,padding=1,bias=False),
            DepthSeperableConv2d(int(512*alpha),int(512*alpha),3,padding=1,bias=False),
            DepthSeperableConv2d(int(512*alpha),int(512*alpha),3,padding=1,bias=False),
            DepthSeperableConv2d(int(512*alpha),int(512*alpha),3,padding=1,bias=False),
        )
        
        self.conv4=nn.Sequential(
            DepthSeperableConv2d(int(512*alpha),int(1024*alpha),3,stride=2,padding=1,bias=False),
            DepthSeperableConv2d(int(1024*alpha),int(1024*alpha),3,padding=1,bias=False),
        )
        
        self.avg=nn.AdaptiveAvgPool2d(1)
        self.fc=nn.Linear(int(1024*alpha),num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
        
    def forward(self,x):
        x=self.stem(x)
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        x=self.avg(x)
        x = torch.flatten(x, 1)
        x=self.fc(x)