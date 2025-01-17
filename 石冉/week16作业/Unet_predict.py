import glob
import cv2
import numpy as np
import torch
from week16_unet_model import UNet

if __name__=='__main__':
    #选择设备，有cuda用cuda，没有用cpu
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #加载unet网络
    net=UNet(n_channels=1,n_classes=1)
    #将网络考到设备中
    net.to(device=device)
    #加载模型参数
    net.load_state_dict(torch.load('best_model.pth',map_location=device))
    #测试
    net.eval()
    #读取所有图片路径
    tests_path=glob.glob('data/test/*.png')
    #遍历所有图片
    for test_path in tests_path:
        #设置结果保存地址
        save_res_path=test_path.split('.')[0]+'_res.png'
        #读取图片，转为灰度图
        img=cv2.imread(test_path)
        img=cv2.cvtColor(img,cv2.COLOR_RGB1GRAY)
        #转换为batch=1，通道=1.大小=512*512的数组
        img=img.reshape(1,1,img.shape[0],img.shape[1])
        #转为tensor
        img_tensor=torch.from_numpy(img)
        #将tensor拷贝到device中
        img_tensor=img_tensor.to(device=device,dtype=torch.float32)

        #进行预测
        pred=net(img_tensor)
        #提取结果
        pred=np.array(pred.data.cpu()[0])[0]

        #处理结果
        pred[pred>=0.5]=255
        pred[pred<0.5]=0
        #保存图片
        cv2.imwrite(save_res_path,pred)

