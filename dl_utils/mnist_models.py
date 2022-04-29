import torch
from robustness.cifar_models import resnet

class MnistResNet(resnet.ResNet):
    def __init__(self, block, num_blocks, num_classes=10, feat_scale=1, wm=1):
        super(MnistResNet, self).__init__(block, num_blocks, num_classes=num_classes)
        self.conv1 = torch.nn.Conv2d(1, 64, 
            kernel_size=3, 
            stride=(1,1), 
            padding=(3,3), bias=False)



def ResNet18(**kwargs):
    return MnistResNet(resnet.BasicBlock, [2,2,2,2], **kwargs)

def ResNet18Wide(**kwargs):
    return MnistResNet(resnet.BasicBlock, [2,2,2,2], wm=5, **kwargs)

def ResNet18Thin(**kwargs):
    return MnistResNet(resnet.BasicBlock, [2,2,2,2], wd=.75, **kwargs)

def ResNet34(**kwargs):
    return MnistResNet(resnet.BasicBlock, [3,4,6,3], **kwargs)

def ResNet50(**kwargs):
    return MnistResNet(resnet.Bottleneck, [3,4,6,3], **kwargs)

def ResNet101(**kwargs):
    return MnistResNet(resnet.Bottleneck, [3,4,23,3], **kwargs)

def ResNet152(**kwargs):
    return MnistResNet(resnet.Bottleneck, [3,8,36,3], **kwargs)

resnet50 = ResNet50
resnet18 = ResNet18
resnet34 = ResNet34
resnet101 = ResNet101
resnet152 = ResNet152
resnet18wide = ResNet18Wide