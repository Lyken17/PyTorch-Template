'''Pre-activation ResNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import init
from torch.autograd import Variable

__all__ = ["PreActResNet",
           "resnet20", "resnet50", "resnet101", "resnet146", "resnet200"]

ReLU_inplace = True


class BasicBlock(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, out_planes, stride=1):
        super(BasicBlock, self).__init__()

        bottle_planes = out_planes // self.expansion
        self.learned = nn.Sequential(
            nn.BatchNorm2d(in_planes),
            nn.ReLU(ReLU_inplace),
            nn.Conv2d(in_planes, bottle_planes, kernel_size=1, stride=stride, padding=0, bias=False),

            nn.BatchNorm2d(bottle_planes),
            nn.ReLU(ReLU_inplace),
            nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3, stride=1, padding=1, bias=False),

            nn.BatchNorm2d(bottle_planes),
            nn.ReLU(ReLU_inplace),
            nn.Conv2d(bottle_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        )

    def forward(self, x):
        # shortcut = self.shortcut(x) if hasattr(self, 'shortcut') else x
        shortcut = x

        learned = self.learned(x)

        res = learned + shortcut
        return res


class Bottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, out_planes, stride=1):
        super(Bottleneck, self).__init__()
        bottle_planes = out_planes // self.expansion
        self.projection = nn.Sequential(
            nn.BatchNorm2d(in_planes),
            nn.ReLU(ReLU_inplace)
        )

        self.learned = nn.Sequential(
            # conv 1x1
            # nn.BatchNorm2d(in_planes),
            # nn.ReLU(ReLU_inplace),
            nn.Conv2d(in_planes, bottle_planes, kernel_size=1, stride=stride, padding=0, bias=False),
            # conv 3x3
            nn.BatchNorm2d(bottle_planes),
            nn.ReLU(ReLU_inplace),
            nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3, stride=1, padding=1, bias=False),
            # conv 1x1
            nn.BatchNorm2d(bottle_planes),
            nn.ReLU(ReLU_inplace),
            nn.Conv2d(bottle_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        )

        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        proj = self.projection(x)

        shortcut = self.shortcut(proj) if hasattr(self, 'shortcut') else proj
        learned = self.learned(proj)

        res = learned + shortcut
        return res


def block(in_planes, out_planes, stride):
    if in_planes != out_planes:
        return Bottleneck(in_planes, out_planes, 2)
    else:
        return BasicBlock(in_planes, out_planes, 1)


class PreActResNet(nn.Module):
    def __init__(self, depth, num_classes=10, in_channels=3):
        super(PreActResNet, self).__init__()
        assert (depth - 2) % 9 == 0, "depth must be 9n+2"
        n = (depth - 2) // 9

        num_blocks = [n, n, n]
        nStages = (16, 64, 128, 256)

        self.conv1 = nn.Conv2d(in_channels, nStages[0], kernel_size=3, stride=1, padding=1, bias=False)

        self.layer1 = self._make_layer(nStages[0], nStages[1], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(nStages[1], nStages[2], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(nStages[2], nStages[3], num_blocks[2], stride=2)

        self.last_bn = nn.BatchNorm2d(nStages[3])
        self.last_relu = nn.ReLU(inplace=True)
        self.linear = nn.Linear(nStages[3], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal(m.weight)
                m.bias.data.zero_()

    def _make_layer(self, nInputPlane, nOutputPlane, count, stride):
        layers = []
        layers.append(Bottleneck(nInputPlane, nOutputPlane, stride))
        for i in range(1, count):
            layers.append(BasicBlock(nOutputPlane, nOutputPlane, 1))
        return nn.Sequential(*layers)

    # def _make_layer(self, block, planes, num_blocks, stride):
    #     strides = [stride] + [1]*(num_blocks-1)
    #     layers = []
    #     for stride in strides:
    #         layers.append(block(self.in_planes, planes, stride))
    #         self.in_planes = planes * block.expansion
    #     return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.last_bn(out)
        out = self.last_relu(out)

        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out


def resnet20(num_classes=100):
    return PreActResNet(20, num_classes)


def resnet56(num_classes=100):
    return PreActResNet(56, num_classes)


def resnet101(num_classes=100):
    return PreActResNet(101, num_classes)


def resnet146(num_classes=100):
    return PreActResNet(146, num_classes)


def resnet200(num_classes=100):
    return PreActResNet(200, num_classes)

def resnet1001(num_classes=100):
    return PreActResNet(1001, num_classes)

def test(model, input):
    output = model(input)
    # print(output, end='')
    print(torch.sum(output).data[0])
    print()


def main():
    torch.manual_seed(0)
    net = PreActResNet(38)
    # d = Variable(torch.ones(1, 3, 32, 32), volatile=True)
    # test(net, d * 0)
    # test(net, d)
    # test(net, d * 1.5)

    total = sum([p.data.nelement() for p in net.parameters()])
    print('  + Number of params: %.2f' % (total / 1e6))


if __name__ == "__main__":
    main()
