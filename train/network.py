import torch
import torch.nn as nn

from network_module import *
#from network_feature_extractor import *

# ----------------------------------------
#         Initialize the networks
# ----------------------------------------
def weights_init(net, init_type = 'normal', init_gain = 0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal
    In our paper, we choose the default setting: zero mean Gaussian distribution with a standard deviation of 0.02
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain = init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a = 0, mode = 'fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain = init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    # apply the initialization function <init_func>
    print('initialize network with %s type' % init_type)
    net.apply(init_func)

# ----------------------------------------
#                Generator
# ----------------------------------------
# Fully convolutional layers in feature extraction part compared to Gray_VGG16_BN
# This is for generator only, so BN cannot be attached to the input and output layers of feature extraction part
# Each output of block (conv*) is "convolutional layer + LeakyReLU" that avoids feature sparse
# We replace the adaptive average pooling layer with a convolutional layer with stride = 2, to ensure the size of feature maps fit classifier

def conv3x3(in_planes, out_planes, stride = 1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size = 3, stride = stride, padding = 1, bias = False)

def conv1x1(in_planes, out_planes, stride = 1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size = 1, stride = stride, bias = False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride = 1, downsample = None, norm_layer = None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.InstanceNorm2d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride = 1, downsample = None, norm_layer = None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.InstanceNorm2d
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.lrelu = nn.LeakyReLU(0.2, inplace = True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.lrelu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.lrelu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes = 1000, zero_init_residual = False, norm_layer = None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.InstanceNorm2d
        self.inplanes = 64
        self.begin = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = 7, stride = 1, padding = 3, bias = False),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(64, 64, kernel_size = 4, stride = 2, padding = 1, bias = False),
            norm_layer(64),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(64, 64, kernel_size = 4, stride = 2, padding = 1, bias = False),
            norm_layer(64),
            nn.LeakyReLU(0.2, inplace = True)
        )

        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer = norm_layer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride = 2, norm_layer = norm_layer)
        self.layer3 = self._make_layer(block, 256, layers[2], stride = 2, norm_layer = norm_layer)
        self.layer4 = self._make_layer(block, 512, layers[3], stride = 2, norm_layer = norm_layer)

        self.last = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.LeakyReLU(0.2, inplace = True)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode = 'fan_out', nonlinearity = 'relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride = 1, norm_layer = None):
        if norm_layer is None:
            norm_layer = nn.InstanceNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer = norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.begin(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.last(x)

        return x

# First stage's generator contains two main Auto-encoder and a global hint network
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class FirstStageNet(nn.Module):
    def __init__(self, opt):
        super(FirstStageNet, self).__init__()
        
        # Feature extractors
        self.fenet = ResNet(Bottleneck, [3, 4, 6, 3])
        self.fenet2 = ResNet(Bottleneck, [3, 4, 6, 3])
        
        # Downsampling path
        self.begin = Conv2dLayer(opt.in_channels, opt.start_channels, 7, 1, 3, pad_type=opt.pad, activation=opt.activ_g, norm='none')
        self.down1 = Conv2dLayer(opt.start_channels, opt.start_channels * 2, 4, 2, 1, pad_type=opt.pad, activation=opt.activ_g, norm=opt.norm)
        self.down2 = Conv2dLayer(opt.start_channels * 2, opt.start_channels * 4, 4, 2, 1, pad_type=opt.pad, activation=opt.activ_g, norm=opt.norm)
        self.down3 = Conv2dLayer(opt.start_channels * 4, opt.start_channels * 8, 4, 2, 1, pad_type=opt.pad, activation=opt.activ_g, norm=opt.norm)
        self.down4 = Conv2dLayer(opt.start_channels * 8, opt.start_channels * 16, 4, 2, 1, pad_type=opt.pad, activation=opt.activ_g, norm=opt.norm)
        self.down5 = Conv2dLayer(opt.start_channels * 16, opt.start_channels * 16, 4, 2, 1, pad_type=opt.pad, activation=opt.activ_g, norm=opt.norm)
        self.down6 = Conv2dLayer(opt.start_channels * 48, opt.start_channels * 16, 4, 2, 1, pad_type=opt.pad, activation=opt.activ_g, norm=opt.norm)
        
        # Transformer block
        transformer_layer = TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        self.transformer = TransformerEncoder(transformer_layer, num_layers=2)
        
        # Upsampling path
        self.up1 = TransposeConv2dLayer(opt.start_channels * 16, opt.start_channels * 16, 3, 1, 1, pad_type=opt.pad, activation=opt.activ_g, norm=opt.norm)
        self.a1 = Self_Attn_FM(opt.start_channels * 16)
        self.up2 = TransposeConv2dLayer(opt.start_channels * 32, opt.start_channels * 16, 3, 1, 1, pad_type=opt.pad, activation=opt.activ_g, norm=opt.norm)
        self.a2 = Self_Attn_FM(opt.start_channels * 16)
        self.up3 = TransposeConv2dLayer(opt.start_channels * 32, opt.start_channels * 8, 3, 1, 1, pad_type=opt.pad, activation=opt.activ_g, norm=opt.norm)
        self.a3 = Self_Attn_FM(opt.start_channels * 8)
        self.up4 = TransposeConv2dLayer(opt.start_channels * 16, opt.start_channels * 4, 3, 1, 1, pad_type=opt.pad, activation=opt.activ_g, norm=opt.norm)
        self.up5 = TransposeConv2dLayer(opt.start_channels * 8, opt.start_channels * 2, 3, 1, 1, pad_type=opt.pad, activation=opt.activ_g, norm=opt.norm)
        self.up6 = TransposeConv2dLayer(opt.start_channels * 4, opt.start_channels, 3, 1, 1, pad_type=opt.pad, activation=opt.activ_g, norm=opt.norm)
        
        # Final layer
        self.final = Conv2dLayer(opt.start_channels * 2, opt.out_channels, 7, 1, 3, pad_type=opt.pad, activation='tanh', norm='none')

    def forward(self, x):
        # Extract luminance channel
        l = x[:, [0], :, :]
        
        # Downsampling
        d0 = self.begin(l)
        d1 = self.down1(d0)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        
        # Global features and bottleneck
        global_features = self.fenet(x)
        global_features2 = self.fenet2(x)
        d5_ = torch.cat((d5, global_features, global_features2), 1)
        d6 = self.down6(d5_)
        
        # Transformer processing
        batch_size = d6.size(0)
        d6 = d6.view(batch_size, 512, -1).permute(0, 2, 1)  # [batch_size, 16, 512]
        d6 = self.transformer(d6)  # [batch_size, 16, 512]
        d6 = d6.permute(0, 2, 1).view(batch_size, 512, 4, 4)  # [batch_size, 512, 4, 4]
        
        # Upsampling with skip connections
        u1 = self.up1(d6)
        u1 = self.a1(u1)
        u1 = torch.cat((u1, d5), 1)
        u2 = self.up2(u1)
        u2 = self.a2(u2)
        u2 = torch.cat((u2, d4), 1)
        u3 = self.up3(u2)
        u3 = self.a3(u3)
        u3 = torch.cat((u3, d3), 1)
        u4 = self.up4(u3)
        u4 = torch.cat((u4, d2), 1)
        u5 = self.up5(u4)
        u5 = torch.cat((u5, d1), 1)
        u6 = self.up6(u5)
        u6 = torch.cat((u6, d0), 1)
        
        # Final output
        x = self.final(u6)
        return x

class SecondStageNet(nn.Module):
    def __init__(self, opt):
        super(SecondStageNet, self).__init__()
        
        # Feature extractors
        self.fenet = ResNet(Bottleneck, [3, 4, 6, 3])
        self.fenet2 = ResNet(Bottleneck, [3, 4, 6, 3])
        
        # Downsampling path
        self.begin = Conv2dLayer(opt.in_channels, opt.start_channels, 7, 1, 3, pad_type=opt.pad, activation=opt.activ_g, norm='none')
        self.down1 = Conv2dLayer(opt.start_channels, opt.start_channels * 2, 4, 2, 1, pad_type=opt.pad, activation=opt.activ_g, norm=opt.norm)
        self.down2 = Conv2dLayer(opt.start_channels * 2, opt.start_channels * 4, 4, 2, 1, pad_type=opt.pad, activation=opt.activ_g, norm=opt.norm)
        self.down3 = Conv2dLayer(opt.start_channels * 4, opt.start_channels * 8, 4, 2, 1, pad_type=opt.pad, activation=opt.activ_g, norm=opt.norm)
        self.down4 = Conv2dLayer(opt.start_channels * 8, opt.start_channels * 16, 4, 2, 1, pad_type=opt.pad, activation=opt.activ_g, norm=opt.norm)
        self.down5 = Conv2dLayer(opt.start_channels * 16, opt.start_channels * 16, 4, 2, 1, pad_type=opt.pad, activation=opt.activ_g, norm=opt.norm)
        self.down6 = Conv2dLayer(opt.start_channels * 48, opt.start_channels * 16, 4, 2, 1, pad_type=opt.pad, activation=opt.activ_g, norm=opt.norm)
        
        # Transformer block
        transformer_layer = TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        self.transformer = TransformerEncoder(transformer_layer, num_layers=2)
        
        # Upsampling path
        self.up1 = TransposeConv2dLayer(opt.start_channels * 16, opt.start_channels * 16, 3, 1, 1, pad_type=opt.pad, activation=opt.activ_g, norm=opt.norm)
        self.a1 = Self_Attn_FM(opt.start_channels * 16)
        self.up2 = TransposeConv2dLayer(opt.start_channels * 32, opt.start_channels * 16, 3, 1, 1, pad_type=opt.pad, activation=opt.activ_g, norm=opt.norm)
        self.a2 = Self_Attn_FM(opt.start_channels * 16)
        self.up3 = TransposeConv2dLayer(opt.start_channels * 32, opt.start_channels * 8, 3, 1, 1, pad_type=opt.pad, activation=opt.activ_g, norm=opt.norm)
        self.a3 = Self_Attn_FM(opt.start_channels * 8)
        self.up4 = TransposeConv2dLayer(opt.start_channels * 16, opt.start_channels * 4, 3, 1, 1, pad_type=opt.pad, activation=opt.activ_g, norm=opt.norm)
        self.up5 = TransposeConv2dLayer(opt.start_channels * 8, opt.start_channels * 2, 3, 1, 1, pad_type=opt.pad, activation=opt.activ_g, norm=opt.norm)
        self.up6 = TransposeConv2dLayer(opt.start_channels * 4, opt.start_channels, 3, 1, 1, pad_type=opt.pad, activation=opt.activ_g, norm=opt.norm)
        
        # Final layer
        self.final = Conv2dLayer(opt.start_channels * 2, opt.out_channels, 7, 1, 3, pad_type=opt.pad, activation='tanh', norm='none')

    def forward(self, x, x_last):
        # Extract luminance channel
        l = x[:, [0], :, :]
        
        # Downsampling
        d0 = self.begin(l)
        d1 = self.down1(d0)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        
        # Global features and bottleneck
        global_features = self.fenet(x)
        global_features2 = self.fenet2(x_last)
        d5_ = torch.cat((d5, global_features, global_features2), 1)
        d6 = self.down6(d5_)
        
        # Transformer processing
        batch_size = d6.size(0)
        d6 = d6.view(batch_size, 512, -1).permute(0, 2, 1)  # [batch_size, 16, 512]
        d6 = self.transformer(d6)  # [batch_size, 16, 512]
        d6 = d6.permute(0, 2, 1).view(batch_size, 512, 4, 4)  # [batch_size, 512, 4, 4]
        
        # Upsampling with skip connections
        u1 = self.up1(d6)
        u1 = self.a1(u1)
        u1 = torch.cat((u1, d5), 1)
        u2 = self.up2(u1)
        u2 = self.a2(u2)
        u2 = torch.cat((u2, d4), 1)
        u3 = self.up3(u2)
        u3 = self.a3(u3)
        u3 = torch.cat((u3, d3), 1)
        u4 = self.up4(u3)
        u4 = torch.cat((u4, d2), 1)
        u5 = self.up5(u4)
        u5 = torch.cat((u5, d1), 1)
        u6 = self.up6(u5)
        u6 = torch.cat((u6, d0), 1)
        
        # Final output
        x = self.final(u6)
        return x

# ----------------------------------------
#               Discriminator
# ----------------------------------------
# This is a kind of PatchGAN. Patch is implied in the output. This is 70 * 70 PatchGAN
class PatchDiscriminator70(nn.Module):
    def __init__(self, opt):
        super(PatchDiscriminator70, self).__init__()
        # Down sampling
        self.block1 = Conv2dLayer(opt.in_channels + opt.out_channels, opt.start_channels * 2, 7, 1, 3, pad_type = opt.pad, activation = opt.activ_g, norm = 'none', sn = True)
        self.block2 = Conv2dLayer(opt.start_channels * 2, opt.start_channels * 4, 4, 2, 1, pad_type = opt.pad, activation = opt.activ_d, norm = opt.norm, sn = True)
        self.block3 = Conv2dLayer(opt.start_channels * 4, opt.start_channels * 8, 4, 2, 1, pad_type = opt.pad, activation = opt.activ_d, norm = opt.norm, sn = True)
        self.block4 = Conv2dLayer(opt.start_channels * 8, opt.start_channels * 16, 4, 2, 1, pad_type = opt.pad, activation = opt.activ_d, norm = opt.norm, sn = True)
        self.a1 = Self_Attn_FM(opt.start_channels * 16)
        # Final output, implemention of 70 * 70 PatchGAN
        self.final1 = Conv2dLayer(opt.start_channels * 16, opt.start_channels * 8, 4, 1, 1, pad_type = opt.pad, activation = opt.activ_d, norm = opt.norm, sn = True)
        self.final2 = Conv2dLayer(opt.start_channels * 8, 1, 4, 1, 1, pad_type = opt.pad, activation = 'none', norm = 'none', sn = True)

    def forward(self, img_A, img_B):
        # img_A: input grayscale image; img_B: generated color image or ground truth image
        img_A = img_A[:, [0], :, :]
        # Concatenate image and condition image by channels to produce input
        x = torch.cat((img_A, img_B), 1)                        # out: batch * 4 * 256 * 256
        x = self.block1(x)                                      # out: batch * 64 * 256 * 256
        x = self.block2(x)                                      # out: batch * 128 * 128 * 128
        x = self.block3(x)                                      # out: batch * 256 * 64 * 64
        x = self.block4(x)                                      # out: batch * 512 * 32 * 32
        x = self.a1(x)
        x = self.final1(x)                                      # out: batch * 512 * 31 * 31
        x = self.final2(x)                                      # out: batch * 512 * 30 * 30
        return x

# ----------------------------------------
#            Perceptual Network
# ----------------------------------------
# VGG-16 conv4_3 features
class PerceptualNet(nn.Module):
    def __init__(self):
        super(PerceptualNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(512, 512, 3, 1, 1)
        )

    def forward(self, x):
        x = self.features(x)
        return x

if __name__ == "__main__":

    import argparse
    
    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    # network parameters
    parser.add_argument('--in_channels', type = int, default = 3, help = 'in channel for U-Net encoder')
    parser.add_argument('--out_channels', type = int, default = 3, help = 'out channel for U-Net decoder')
    parser.add_argument('--pad', type = str, default = 'zero', help = 'padding type')
    parser.add_argument('--activ_g', type = str, default = 'relu', help = 'activation function for generator')
    parser.add_argument('--activ_d', type = str, default = 'lrelu', help = 'activation function for discriminator')
    parser.add_argument('--norm_g', type = str, default = 'in', help = 'normalization type for generator')
    parser.add_argument('--norm_d', type = str, default = 'in', help = 'normalization type for discriminator')
    parser.add_argument('--init_type', type = str, default = 'normal', help = 'intialization type for generator and discriminator')
    parser.add_argument('--init_gain', type = float, default = 0.02, help = 'the standard deviation if Gaussian normalization')
    opt = parser.parse_args()

    net = FirstStageNet(opt).cuda()
    torch.save(net.state_dict(), 'test.pth')
    a = torch.randn(1, 3, 256, 256).cuda()
    b = net(a)
    loss = torch.mean(b)
    print(b.shape)
    loss.backward()
