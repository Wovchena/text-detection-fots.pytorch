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
    # def __init__(self, in_channels, squeeze_channels):
    #     super().__init__()
    #     self.squeeze = conv(in_channels, squeeze_channels)
    #
    # def forward(self, x, encoder_features):
    #     x = self.squeeze(x)
    #     x = F.interpolate(x, size=(encoder_features.shape[2], encoder_features.shape[3]),
    #                       mode='bilinear', align_corners=True)
    #     up = torch.cat([encoder_features, x], 1)
    #     return up


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
        self.encoder2 = self.resnet.layer2  # 128
        self.encoder3 = self.resnet.layer3  # 256
        self.encoder4 = self.resnet.layer4  # 512

        # self.encoder1 = self.resnet.layer1  # 64
        # self.encoder2 = nn.Sequential(self.resnet.layer2[0:3], ChannelSpatialSELayer(128), self.resnet.layer2[3])  # 128
        # self.encoder3 = nn.Sequential(self.resnet.layer3[0:5], ChannelSpatialSELayer(256), self.resnet.layer3[5])  # 256
        # self.encoder4 = self.encoder4 = nn.Sequential(self.resnet.layer4, ChannelSpatialSELayer(512))  # 512

        self.sAttnE3 = Self_Attn(256)
        self.sAttnE4 = Self_Attn(512)

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

        ae3 = self.sAttnE3(e3)
        ae4 = self.sAttnE4(e4)

        d4 = self.decoder4(ae4, ae3)
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

#
#
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision
#
#
# class Self_Attn(nn.Module):
#     """ Self attention Layer"""
#
#     def __init__(self, in_dim):
#         super(Self_Attn, self).__init__()
#         self.chanel_in = in_dim
#
#         self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
#         self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
#         self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
#         self.gamma = nn.Parameter(torch.zeros(1))
#
#         self.softmax = nn.Softmax(dim=-1)  #
#
#     def forward(self, x):
#         """
#             inputs :
#                 x : input feature maps( B X C X W X H)
#             returns :
#                 out : self attention value + input feature
#                 attention: B X N X N (N is Width*Height)
#         """
#         m_batchsize, C, width, height = x.size()
#         proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
#         proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
#         energy = torch.bmm(proj_query, proj_key)  # transpose check
#         attention = self.softmax(energy)  # BX (N) X (N)
#         proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N
#
#         out = torch.bmm(proj_value, attention.permute(0, 2, 1))
#         out = out.view(m_batchsize, C, width, height)
#
#         out = self.gamma * out + x
#         return out#, attention
#
# class ChannelSELayer(nn.Module):
#     """
#     Re-implementation of Squeeze-and-Excitation (SE) block described in:
#         *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
#     """
#
#     def __init__(self, num_channels, reduction_ratio=2):
#         """
#         :param num_channels: No of input channels
#         :param reduction_ratio: By how much should the num_channels should be reduced
#         """
#         super(ChannelSELayer, self).__init__()
#         num_channels_reduced = num_channels // reduction_ratio
#         self.reduction_ratio = reduction_ratio
#         self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
#         self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
#         self.relu = nn.ReLU()
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, input_tensor):
#         """
#         :param input_tensor: X, shape = (batch_size, num_channels, H, W)
#         :return: output tensor
#         """
#         batch_size, num_channels, H, W = input_tensor.size()
#         # Average along each channel
#         squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)
#
#         # channel excitation
#         fc_out_1 = self.relu(self.fc1(squeeze_tensor))
#         fc_out_2 = self.sigmoid(self.fc2(fc_out_1))
#
#         a, b = squeeze_tensor.size()
#         output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1))
#         return output_tensor
#
#
# class SpatialSELayer(nn.Module):
#     """
#     Re-implementation of SE block -- squeezing spatially and exciting channel-wise described in:
#         *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, MICCAI 2018*
#     """
#
#     def __init__(self, num_channels):
#         """
#         :param num_channels: No of input channels
#         """
#         super(SpatialSELayer, self).__init__()
#         self.conv = nn.Conv2d(num_channels, 1, 1)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, input_tensor, weights=None):
#         """
#         :param weights: weights for few shot learning
#         :param input_tensor: X, shape = (batch_size, num_channels, H, W)
#         :return: output_tensor
#         """
#         # spatial squeeze
#         batch_size, channel, a, b = input_tensor.size()
#
#         if weights is not None:
#             weights = torch.mean(weights, dim=0)
#             weights = weights.view(1, channel, 1, 1)
#             out = F.conv2d(input_tensor, weights)
#         else:
#             out = self.conv(input_tensor)
#         squeeze_tensor = self.sigmoid(out)
#
#         # spatial excitation
#         # print(input_tensor.size(), squeeze_tensor.size())
#         squeeze_tensor = squeeze_tensor.view(batch_size, 1, a, b)
#         output_tensor = torch.mul(input_tensor, squeeze_tensor)
#         #output_tensor = torch.mul(input_tensor, squeeze_tensor)
#         return output_tensor
#
#
# class ChannelSpatialSELayer(nn.Module):
#     """
#     Re-implementation of concurrent spatial and channel squeeze & excitation:
#         *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, MICCAI 2018, arXiv:1803.02579*
#     """
#
#     def __init__(self, num_channels, reduction_ratio=2):
#         """
#         :param num_channels: No of input channels
#         :param reduction_ratio: By how much should the num_channels should be reduced
#         """
#         super(ChannelSpatialSELayer, self).__init__()
#         self.cSE = ChannelSELayer(num_channels, reduction_ratio)
#         self.sSE = SpatialSELayer(num_channels)
#
#     def forward(self, input_tensor):
#         """
#         :param input_tensor: X, shape = (batch_size, num_channels, H, W)
#         :return: output_tensor
#         """
#         output_tensor = torch.add(self.cSE(input_tensor), self.sSE(input_tensor))
#         return output_tensor
#
# def conv(in_channels, out_channels, kernel_size=3, padding=1, bn=True, dilation=1, stride=1, relu=True, bias=True):
#     modules = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)]
#     if bn:
#         modules.append(nn.BatchNorm2d(out_channels))
#     if relu:
#         modules.append(nn.ReLU(inplace=True))
#     return nn.Sequential(*modules)
#
#
# # class Decoder(nn.Module):
# #     def __init__(self, in_channels, squeeze_channels):
# #         super().__init__()
# #         self.squeeze = conv(in_channels, squeeze_channels)
# #
# #     def forward(self, x, encoder_features):
# #         x = self.squeeze(x)
# #         x = F.interpolate(x, size=(encoder_features.shape[2], encoder_features.shape[3]),
# #                           mode='bilinear', align_corners=True)
# #         up = torch.cat([encoder_features, x], 1)
# #         return up
#
#
# class Decoder(nn.Module):
#     def __init__(self, in_channels, encoder_channels):
#         super().__init__()
#         out_channels = (in_channels + encoder_channels) // 2
#         self.conv1 = conv(in_channels + encoder_channels, out_channels)
#         self.conv2 = conv(out_channels, out_channels)
#
#     def forward(self, x, encoder_features):
#         x = F.interpolate(x, size=(encoder_features.shape[2], encoder_features.shape[3]),
#                           mode='bilinear', align_corners=True)
#         up = self.conv1(torch.cat([encoder_features, x], 1))
#         up = self.conv2(up)
#         return up
#
# import torch.nn as nn
# import torch.utils.model_zoo as model_zoo
# from torchvision.models.resnet import conv1x1, conv3x3, BasicBlock, Bottleneck
#
# class ResNet(nn.Module):
#
#     def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
#         super(ResNet, self).__init__()
#         self.inplanes = 64
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
#                                bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(block, 64, layers[0])
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=1)
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=1)
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(512 * block.expansion, num_classes)
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#
#         # Zero-initialize the last BN in each residual branch,
#         # so that the residual branch starts with zeros, and each residual block behaves like an identity.
#         # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
#         if zero_init_residual:
#             for m in self.modules():
#                 if isinstance(m, Bottleneck):
#                     nn.init.constant_(m.bn3.weight, 0)
#                 elif isinstance(m, BasicBlock):
#                     nn.init.constant_(m.bn2.weight, 0)
#
#     def _make_layer(self, block, planes, blocks, stride=1):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 conv1x1(self.inplanes, planes * block.expansion, stride),
#                 nn.BatchNorm2d(planes * block.expansion),
#             )
#
#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample))
#         self.inplanes = planes * block.expansion
#         for _ in range(1, blocks):
#             layers.append(block(self.inplanes, planes))
#
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#
#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#
#         return x
#
# def resnet34(pretrained=False, **kwargs):
#     """Constructs a ResNet-34 model.
#
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
#     if pretrained:
#         model_urls = {
#             'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
#             'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
#             'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
#             'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
#             'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
#         }
#         model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
#     return model
#
# class FOTSModel(nn.Module):
#     def __init__(self, crop_height=640):
#         super().__init__()
#         self.crop_height = crop_height
#         self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
#
#         self.resnet = resnet34(pretrained=True)
#         self.conv1 = nn.Sequential(
#             self.resnet.conv1,
#             self.resnet.bn1,
#             self.resnet.relu,
#         )  # 64
#
#         self.encoder1 = nn.Sequential(self.resnet.layer1[0:3])  # 64
#         self.encoder2 = nn.Sequential(self.resnet.layer2[0:4])  # 128
#         self.encoder3 = nn.Sequential(self.resnet.layer3[0:5])  # 256
#
#         self.remove_artifacts = conv(256, 64)
#
#         self.confidence = conv(64, 1, kernel_size=1, padding=0, bn=False, relu=False)
#         self.distances = conv(64, 4, kernel_size=1, padding=0, bn=False, relu=False)
#         self.angle = conv(64, 1, kernel_size=1, padding=0, bn=False, relu=False)
#
#     def forward(self, x):
#         e0 = self.conv1(x)
#         e0 = self.pool1(e0)
#
#         e1 = self.encoder1(e0)
#         e2 = self.encoder2(e1)
#         e3 = self.encoder3(e2)
#
#         final = self.remove_artifacts(e3)
#
#         confidence = self.confidence(final)
#         confidence = torch.sigmoid(confidence)
#         distances = self.distances(final)
#         distances = torch.sigmoid(distances) * (self.crop_height / 2)
#         angle = self.angle(final)
#         angle = torch.sigmoid(angle) * np.pi / 2
#
#         return confidence, distances, angle
