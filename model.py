import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out#, attention

class ChannelSELayer(nn.Module):
    """
    Re-implementation of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
    """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output tensor
        """
        batch_size, num_channels, H, W = input_tensor.size()
        # Average along each channel
        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1))
        return output_tensor


class SpatialSELayer(nn.Module):
    """
    Re-implementation of SE block -- squeezing spatially and exciting channel-wise described in:
        *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, MICCAI 2018*
    """

    def __init__(self, num_channels):
        """
        :param num_channels: No of input channels
        """
        super(SpatialSELayer, self).__init__()
        self.conv = nn.Conv2d(num_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor, weights=None):
        """
        :param weights: weights for few shot learning
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output_tensor
        """
        # spatial squeeze
        batch_size, channel, a, b = input_tensor.size()

        if weights is not None:
            weights = torch.mean(weights, dim=0)
            weights = weights.view(1, channel, 1, 1)
            out = F.conv2d(input_tensor, weights)
        else:
            out = self.conv(input_tensor)
        squeeze_tensor = self.sigmoid(out)

        # spatial excitation
        # print(input_tensor.size(), squeeze_tensor.size())
        squeeze_tensor = squeeze_tensor.view(batch_size, 1, a, b)
        output_tensor = torch.mul(input_tensor, squeeze_tensor)
        #output_tensor = torch.mul(input_tensor, squeeze_tensor)
        return output_tensor


class ChannelSpatialSELayer(nn.Module):
    """
    Re-implementation of concurrent spatial and channel squeeze & excitation:
        *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, MICCAI 2018, arXiv:1803.02579*
    """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSpatialSELayer, self).__init__()
        self.cSE = ChannelSELayer(num_channels, reduction_ratio)
        self.sSE = SpatialSELayer(num_channels)

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output_tensor
        """
        output_tensor = torch.add(self.cSE(input_tensor), self.sSE(input_tensor))
        return output_tensor

def conv(in_channels, out_channels, kernel_size=3, padding=1, bn=True, dilation=1, stride=1, relu=True, bias=True):
    modules = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)]
    if bn:
        modules.append(nn.BatchNorm2d(out_channels))
    if relu:
        modules.append(nn.ReLU(inplace=True))
    return nn.Sequential(*modules)


# class Decoder(nn.Module):
#     def __init__(self, in_channels, squeeze_channels):
#         super().__init__()
#         self.squeeze = conv(in_channels, squeeze_channels)
#
#     def forward(self, x, encoder_features):
#         x = self.squeeze(x)
#         x = F.interpolate(x, size=(encoder_features.shape[2], encoder_features.shape[3]),
#                           mode='bilinear', align_corners=True)
#         up = torch.cat([encoder_features, x], 1)
#         return up


class Decoder(nn.Module):
    def __init__(self, in_channels, encoder_channels):
        super().__init__()
        out_channels = (in_channels + encoder_channels) // 2
        self.conv1 = conv(in_channels + encoder_channels, out_channels)
        self.conv2 = conv(out_channels, out_channels)

    def forward(self, x, encoder_features):
        x = F.interpolate(x, size=(encoder_features.shape[2], encoder_features.shape[3]),
                          mode='bilinear', align_corners=True)
        up = self.conv1(torch.cat([encoder_features, x], 1))
        up = self.conv2(up)
        return up


class FOTSModel(nn.Module):
    def __init__(self, crop_height=640):
        super().__init__()
        self.crop_height = crop_height
        self.resnet = torchvision.models.resnet34(pretrained=True)
        self.conv1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
        )  # 64

        self.encoder1 = self.resnet.layer1  # 64
        self.encoder2 = nn.Sequential(self.resnet.layer2[0:3], self.resnet.layer2[3])  # 128
        self.encoder3 = nn.Sequential(self.resnet.layer3[0:5], self.resnet.layer3[5])  # 256
        self.encoder4 = nn.Sequential(self.resnet.layer4)  # 512

        # self.encoder1 = self.resnet.layer1  # 64
        # self.encoder2 = nn.Sequential(self.resnet.layer2[0:3], ChannelSpatialSELayer(128), self.resnet.layer2[3])  # 128
        # self.encoder3 = nn.Sequential(self.resnet.layer3[0:5], ChannelSpatialSELayer(256), self.resnet.layer3[5])  # 256
        # self.encoder4 = self.encoder4 = nn.Sequential(self.resnet.layer4, ChannelSpatialSELayer(512))  # 512

        self.decoder4 = Decoder(512, 256)
        self.decoder3 = Decoder(384, 128)
        self.decoder2 = Decoder(256, 64)
        self.decoder1 = Decoder(160, 64)
        self.remove_artifacts = conv(112, 64)

        self.confidence = conv(64, 1, kernel_size=1, padding=0, bn=False, relu=False)
        self.distances = conv(64, 4, kernel_size=1, padding=0, bn=False, relu=False)
        self.angle = conv(64, 1, kernel_size=1, padding=0, bn=False, relu=False)

    def forward(self, x):
        e0 = self.conv1(x)
        e0_pooled = F.max_pool2d(e0, kernel_size=2, stride=2)

        e1 = self.encoder1(e0_pooled)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        d4 = self.decoder4(e4, e3)
        d3 = self.decoder3(d4, e2)
        d2 = self.decoder2(d3, e1)
        d1 = self.decoder1(d2, e0)

        final = self.remove_artifacts(d1)

        confidence = self.confidence(final)
        confidence = torch.sigmoid(confidence)
        distances = self.distances(final)
        distances = torch.sigmoid(distances) * (self.crop_height / 2)
        angle = self.angle(final)
        angle = torch.sigmoid(angle) * np.pi / 2

        return confidence, distances, angle
