
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import torch
from torch import nn
import torch.nn.functional as F
# import os
# from torch.utils.tensorboard import SummaryWriter

class ConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none', activation=nn.ReLU, dilate=None):
        super(ConvBlock, self).__init__()

        if dilate == None:
            dilate = [1] * n_stages

        if len(dilate) != n_stages:
            raise Exception("len() of dilate num doesn't match!")

        ops = []
        for i in range(n_stages):
            if i==0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=dilate[i], dilation=dilate[i]))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=4, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False
            ops.append(activation(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class ResidualConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none', activation=nn.ReLU, dilate=None):
        super(ResidualConvBlock, self).__init__()

        if dilate == None:
            dilate = [1] * n_stages

        if len(dilate) != n_stages:
            raise Exception("len() of dilate num doesn't match!")

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=dilate[i], dilation=dilate[i]))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=4, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False

            if i != n_stages-1:
                ops.append(activation(inplace=True))

        self.conv = nn.Sequential(*ops)
        self.relu = activation(inplace=True)

    def forward(self, x):
        x = (self.conv(x) + x)
        x = self.relu(x)
        return x


class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none', activation=nn.ReLU):
        super(DownsamplingConvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=4, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(activation(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class UpsamplingDeconvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none', activation=nn.ReLU):
        super(UpsamplingDeconvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=4, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(activation(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class Upsampling(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none', activation=nn.ReLU):
        super(Upsampling, self).__init__()

        ops = []
        ops.append(nn.Upsample(scale_factor=stride, mode='trilinear',align_corners=False))
        ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))
        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=4, num_channels=n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        elif normalization != 'none':
            assert False
        ops.append(activation(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x

class CardiacSeg(nn.Module):
    def __init__(self):
        super().__init__()

        self.vnet = VNet(n_channels=1, n_classes=3, normalization='instancenorm', has_dropout=False, activation=nn.ReLU)
        self.loss1 = DiceLoss()
        self.loss2 = nn.CrossEntropyLoss()

    def forward(self, x, label=None):
        out = self.vnet(x)

        if type(label) == torch.Tensor:
            loss1 = self.loss1(F.softmax(out, dim=1), label)
            # loss2 = self.loss2(out, label.argmax(dim=1))

            loss = loss1
            return out.detach(), loss
        else:
            return (F.softmax(out, dim=1)).detach()

class DUC_UpSampler(nn.Module):

    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none', activation=nn.ReLU):
        super().__init__()

        ops = []
        ops.append(nn.Conv3d(n_filters_in, n_filters_in * (stride ** 3), stride=1, kernel_size=1, padding=0))
        ops.append(PixelShuffle3D(stride))
        ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))
        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=4, num_channels=n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        elif normalization != 'none':
            assert False
        ops.append(activation(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        out = self.conv(x)

        return out

class PixelShuffle3D(nn.Module):

    def __init__(self, upscale_factor):
        super(PixelShuffle3D, self).__init__()

        self.upscale_factor = upscale_factor

    def forward(self, inputs):

        batch_size, channels, in_depth, in_height, in_width = inputs.size()

        channels //= self.upscale_factor ** 3

        out_depth = in_depth * self.upscale_factor
        out_height = in_height * self.upscale_factor
        out_width = in_width * self.upscale_factor

        input_view = inputs.contiguous().view(
            batch_size, channels, self.upscale_factor, self.upscale_factor, self.upscale_factor,
            in_depth, in_height, in_width)

        shuffle_out = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()

        return shuffle_out.view(batch_size, channels, out_depth, out_height, out_width)

class SCAttention(nn.Module):

    def __init__(self, inplane, ratio=4, groups=2):
        super().__init__()

        self.channelatt = nn.Sequential(
            nn.Conv3d(inplane, inplane // ratio, 1, 1, groups=groups),
            nn.Tanh(),
            nn.Conv3d(inplane // ratio, inplane, 1, 1, groups=groups)
        )
        
        self.spatialatt = nn.Sequential(
            nn.Conv3d(inplane, inplane // 2, 1, 1, 0, groups=groups),
            nn.InstanceNorm3d(inplane // 2),
            nn.Tanh(),
            nn.Conv3d(inplane // 2, groups, 7, 1, 3, groups=groups),
            nn.InstanceNorm3d(groups)
        )

        self.groups = groups
        self.ch_pergroup = inplane // groups

    def forward(self, x):
        channelatt_sum = F.adaptive_max_pool3d(x, 1) + F.adaptive_avg_pool3d(x, 1)
        channelatt_sum = self.channelatt(channelatt_sum)
        channelatt = torch.sigmoid(channelatt_sum).expand_as(x)

        x1 = x * channelatt
        N, C, Z, Y, X = x1.size()

        spatialatt_c = self.spatialatt(x1)
        spatialatt_c = torch.sigmoid(spatialatt_c)

        spatialatt_c = spatialatt_c.unsqueeze(dim=2).expand(-1, -1, self.ch_pergroup, -1, -1, -1)
        spatialatt_c = spatialatt_c.reshape(N, C, Z, Y, X)
    
        x1 = x1 * spatialatt_c

        return x + x1

class VNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False, activation=nn.ReLU):
        super(VNet, self).__init__()
        self.has_dropout = has_dropout

        self.block_one = ConvBlock(1, n_channels, n_filters * 1, normalization=normalization, activation=activation)
        self.block_one_dw = DownsamplingConvBlock(n_filters * 1, n_filters * 2, normalization=normalization, activation=activation, stride=(1,2,2))
        
        self.block_22 = ResidualConvBlock(3, n_filters * 2, n_filters * 2, normalization=normalization, activation=activation, dilate=[1,2,5])

        self.block_32 = ResidualConvBlock(3, n_filters * 2, n_filters * 2, normalization=normalization, activation=activation, dilate=[1,2,5])
    
        self.block_42 = ConvBlock(3, n_filters * 2, n_filters * 1, normalization=normalization, activation=activation, dilate=[1,2,5])
      
        self.block_eight_up = UpsamplingDeconvBlock(n_filters * 1, n_filters * 1, normalization=normalization, activation=activation, stride=(1,2,2))
        self.attention_four = SCAttention(n_filters * 1)

        self.block_nine = ConvBlock(1, n_filters * 1, n_filters, normalization=normalization, activation=activation)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)

        self.__init_weight()

    def encoder(self, input):
        x1 = self.block_one(input)
        x2 = self.block_one_dw(x1)

        x2 = self.block_22(x2)

        x2 = self.block_32(x2)

        x2 = self.block_42(x2)

        res = [x1, x2]

        return res

    def decoder(self, features):
        x1 = features[0]
        x2 = features[1]

        x2_up = self.block_eight_up(x2)
        x = self.attention_four(x1 + x2_up)
 
        x = self.block_nine(x)
    
        out = self.out_conv(x)

        return out


    def forward(self, input, turnoff_drop=False):
        features = self.encoder(input)
        out = self.decoder(features)

        return out

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


# if __name__ == "__main__":
#     writer = SummaryWriter(os.path.join('src', 'log', 'tensorboard'))
#     model = VNet(n_channels=1, n_classes=3, normalization='instancenorm', has_dropout=False, activation=nn.ReLU).cuda()
#     tin = torch.zeros(1, 1, 128, 128, 128).cuda()
    
#     writer.add_graph(model, tin)
#     writer.close()  