import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision


def mean_image_subtraction(images, means=[123.68, 116.78, 103.94]):
    '''
    image normalization
    :param images: bs * w * h * channel
    :param means:
    :return:
    '''
    num_channels = images.data.shape[1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')
    for i in range(num_channels):
        images.data[:, i, :, :] -= means[i]

    return images


class DummyLayer(nn.Module):
    def __init__(self):
        super(DummyLayer, self).__init__()

    def forward(self, input_f):
        return input_f


class HLayer(nn.Module):
    def __init__(self, inputChannels, outputChannels):
        """

        :param inputChannels: channels of g+f
        :param outputChannels:
        """
        super(HLayer, self).__init__()

        self.conv2dOne = nn.Conv2d(inputChannels, outputChannels, kernel_size=1)
        self.bnOne = nn.BatchNorm2d(outputChannels, momentum=0.003)

        self.conv2dTwo = nn.Conv2d(outputChannels, outputChannels, kernel_size=3, padding=1)
        self.bnTwo = nn.BatchNorm2d(outputChannels, momentum=0.003)

    def forward(self, inputPrevG, inputF):
        input = torch.cat([inputPrevG, inputF], dim=1)
        output = self.conv2dOne(input)
        output = self.bnOne(output)
        output = F.relu(output)

        output = self.conv2dTwo(output)
        output = self.bnTwo(output)
        output = F.relu(output)

        return output


class FOTSModel(nn.Module):
    def __init__(self):
        super(FOTSModel, self).__init__()

        self.backbone = torchvision.models.resnet50(pretrained=True)

        self.mergeLayers0 = DummyLayer()

        self.mergeLayers1 = HLayer(2048 + 1024, 128)
        self.mergeLayers2 = HLayer(128 + 512, 64)
        self.mergeLayers3 = HLayer(64 + 256, 32)

        self.mergeLayers4 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(32, momentum=0.003)

        # Output Layers
        self.scoreMap = nn.Conv2d(32, 1, kernel_size=1)
        self.geoMap = nn.Conv2d(32, 4, kernel_size=1)
        self.angleMap = nn.Conv2d(32, 1, kernel_size=1)

    def foward_backbone(self, input):
        conv2 = None
        conv3 = None
        conv4 = None
        output = None

        for name, layer in self.backbone.named_children():
            input = layer(input)
            if name == 'layer1':
                conv2 = input
            elif name == 'layer2':
                conv3 = input
            elif name == 'layer3':
                conv4 = input
            elif name == 'layer4':
                output = input
                break

        return output, conv4, conv3, conv2

    def unpool(self, input):
        return F.interpolate(input, mode='bilinear', scale_factor=2, align_corners=True)

    def forward(self, input):
        # input = self.mean_image_subtraction(input)
        f = self.foward_backbone(input)

        g = [None] * 4
        h = [None] * 4

        # i = 1
        h[0] = self.mergeLayers0(f[0])
        g[0] = self.unpool(h[0])

        # i = 2
        h[1] = self.mergeLayers1(g[0], f[1])
        g[1] = self.unpool(h[1])

        # i = 3
        h[2] = self.mergeLayers2(g[1], f[2])
        g[2] = self.unpool(h[2])

        # i = 4
        h[3] = self.mergeLayers3(g[2], f[3])

        # final stage
        final = self.mergeLayers4(h[3])
        final = self.bn5(final)
        final = F.relu(final)

        score = self.scoreMap(final)
        score = torch.sigmoid(score)

        geoMap = self.geoMap(final)
        geoMap = torch.sigmoid(geoMap) * geoMap.shape[2]
        angleMap = self.angleMap(final)
        angleMap = torch.sigmoid(angleMap) * -90  # TODO do I really need sigmoid

        return score, geoMap, angleMap
