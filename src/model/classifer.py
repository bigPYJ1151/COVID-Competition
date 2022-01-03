import torch
from torch import nn
from torch.functional import Tensor
import torch.nn.functional as F

from torch.utils.tensorboard.writer import SummaryWriter

class ConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none', activation=nn.ReLU):
        super(ConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i==0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
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
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none', activation=nn.ReLU):
        super(ResidualConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
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

class ResVNetv2(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False, activation=nn.ReLU):
        super(ResVNetv2, self).__init__()
        self.has_dropout = has_dropout

        self.block_one = ResidualConvBlock(1, n_channels, n_filters, normalization=normalization, activation=activation)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization, activation=activation)

        self.block_two = ResidualConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization, activation=activation)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization, activation=activation)

        self.block_three = ResidualConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization, activation=activation)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization, activation=activation)

        self.block_four = ResidualConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization, activation=activation)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization, activation=activation)

        self.block_five = ResidualConvBlock(3, n_filters * 16, n_filters * 16, normalization=normalization, activation=activation)
        self.block_five_up = UpsamplingDeconvBlock(n_filters * 16, n_filters * 8, normalization=normalization, activation=activation)
        self.attention_one = SCAttention(n_filters * 8)

        self.block_six = ResidualConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization, activation=activation)
        self.block_six_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4, normalization=normalization, activation=activation)
        self.attention_two = SCAttention(n_filters * 4)

        self.block_seven = ResidualConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization, activation=activation)
        self.block_seven_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, normalization=normalization, activation=activation)
        self.attention_three = SCAttention(n_filters * 2)

        self.block_eight = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization, activation=activation)
        self.block_eight_up = DUC_UpSampler(n_filters * 2, n_filters, normalization=normalization, activation=activation)
        self.attention_four = SCAttention(n_filters)

        self.block_nine = ConvBlock(1, n_filters, n_filters, normalization=normalization, activation=activation)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)

        self.__init_weight()

    def encoder(self, input):
        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)

        res = [x1, x2, x3, x4, x5]

        return res

    def decoder(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        x5_up = self.block_five_up(x5)
        x5_up = self.attention_one(x4 + x5_up)
        # x5_up = x5_up + x4_a

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = self.attention_two(x3 + x6_up)
        # x6_up = x6_up + x3_a

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = self.attention_three(x2 + x7_up)
        # x7_up = x7_up + x2_a

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = self.attention_four(x1 + x8_up)
        # x8_up = x8_up + x1_a
        x9 = self.block_nine(x8_up)
        # x9 = F.dropout3d(x9, p=0.5, training=True)

        out = self.out_conv(x9)
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

class ResVNetv1(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False, activation=nn.ReLU):
        super(ResVNetv1, self).__init__()
        self.has_dropout = has_dropout

        self.block_one = ConvBlock(1, n_channels, n_filters, normalization=normalization, activation=activation)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization, activation=activation)

        self.block_two = ResidualConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization, activation=activation)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization, activation=activation)

        self.block_three = ResidualConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization, activation=activation)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization, activation=activation)

        self.block_four = ResidualConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization, activation=activation)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization, activation=activation)

        self.block_five = ConvBlock(3, n_filters * 16, n_filters * 16, normalization=normalization, activation=activation)
        self.block_five_up = UpsamplingDeconvBlock(n_filters * 16, n_filters * 8, normalization=normalization, activation=activation)
        self.attention_one = SCAttention(n_filters * 8)

        self.block_six = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization, activation=activation)
        self.block_six_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4, normalization=normalization, activation=activation)
        self.attention_two = SCAttention(n_filters * 4)

        self.block_seven = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization, activation=activation)
        self.block_seven_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, normalization=normalization, activation=activation)
        self.attention_three = SCAttention(n_filters * 2)

        self.block_eight = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization, activation=activation)
        self.block_eight_up = DUC_UpSampler(n_filters * 2, n_filters, normalization=normalization, activation=activation)
        self.attention_four = SCAttention(n_filters)

        self.block_nine = ConvBlock(1, n_filters, n_filters, normalization=normalization, activation=activation)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)

        self.__init_weight()

    def encoder(self, input):
        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)

        res = [x1, x2, x3, x4, x5]

        return res

    def decoder(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        x5_up = self.block_five_up(x5)
        x5_up = self.attention_one(x4 + x5_up)
        # x5_up = x5_up + x4_a

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = self.attention_two(x3 + x6_up)
        # x6_up = x6_up + x3_a

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = self.attention_three(x2 + x7_up)
        # x7_up = x7_up + x2_a

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = self.attention_four(x1 + x8_up)
        # x8_up = x8_up + x1_a
        x9 = self.block_nine(x8_up)
        # x9 = F.dropout3d(x9, p=0.5, training=True)

        out = self.out_conv(x9)
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

class VNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False, activation=nn.ReLU):
        super(VNet, self).__init__()
        self.has_dropout = has_dropout
        self.scale_factor = 1.0

        self.block_one = ConvBlock(1, n_channels, n_filters, normalization=normalization, activation=activation)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization, activation=activation)
        self.scale_factor = self.scale_factor * 2

        self.block_two = ResidualConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization, activation=activation)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization, activation=activation)
        self.scale_factor = self.scale_factor * 2

        self.block_three = ResidualConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization, activation=activation)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization, activation=activation)
        self.scale_factor = self.scale_factor * 2

        self.block_four = ResidualConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization, activation=activation)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization, activation=activation)
        self.scale_factor = self.scale_factor * 2

        self.block_five = ResidualConvBlock(3, n_filters * 16, n_filters * 16, normalization=normalization, activation=activation)

        self.out_gap = nn.AdaptiveAvgPool3d(1)
        self.output_conv = nn.Conv3d(n_filters * 16, 2, 1)
        
        self.drop = nn.Dropout3d()

        self.__init_weight()

    def encoder(self, input):
        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)

        return x5

    def forward(self, input, turnoff_drop=False):
        features = self.encoder(input)

        out = self.out_gap(features)
        out = self.output_conv(out)

        out = self.drop(out)

        N, C, _, _, _ = out.size()
        out = out.view(N, C)

        cam = self.getCAM(features)
        cam = F.interpolate(cam, scale_factor=self.scale_factor, mode='trilinear', align_corners=True)

        return out, cam

    def getCAM(self, prevOutput : Tensor):
        N, C, D, H, W = prevOutput.size()
        prevOutput = prevOutput.view(N, C, -1)

        outWeights = self.output_conv.weight.squeeze(2).squeeze(2).squeeze(2)
        KOUT, KIN = outWeights.size()
        outWeights = outWeights.unsqueeze(0)
        outWeights = outWeights.expand(N, KOUT, KIN)

        return torch.bmm(outWeights, prevOutput).view(N, KOUT, D, H, W)

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class VNet_Upsampling(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False, activation=nn.ReLU):
        super(VNet_Upsampling, self).__init__()
        self.has_dropout = has_dropout

        self.block_one = ConvBlock(1, n_channels, n_filters, normalization=normalization, activation=activation)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization, activation=activation)

        self.block_two = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization, activation=activation)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization, activation=activation)

        self.block_three = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization, activation=activation)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization, activation=activation)

        self.block_four = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization, activation=activation)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization, activation=activation)

        self.block_five = ConvBlock(3, n_filters * 16, n_filters * 16, normalization=normalization, activation=activation)
        self.block_five_up = DUC_UpSampler(n_filters * 16, n_filters * 8, normalization=normalization, activation=activation)
        self.attention_one = SCAttention(n_filters * 8)

        self.block_six = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization, activation=activation)
        self.block_six_up = DUC_UpSampler(n_filters * 8, n_filters * 4, normalization=normalization, activation=activation)
        self.attention_two = SCAttention(n_filters * 4)

        self.block_seven = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization, activation=activation)
        self.block_seven_up = DUC_UpSampler(n_filters * 4, n_filters * 2, normalization=normalization, activation=activation)
        self.attention_three = SCAttention(n_filters * 2)

        self.block_eight = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization, activation=activation)
        self.block_eight_up = Upsampling(n_filters * 2, n_filters, normalization=normalization, activation=activation)
        self.attention_four = SCAttention(n_filters)

        self.block_nine = ConvBlock(1, n_filters, n_filters, normalization=normalization, activation=activation)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)

        self.__init_weight()

    def encoder(self, input):
        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)

        res = [x1, x2, x3, x4, x5]

        return res

    def decoder(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        x5_up = self.block_five_up(x5)
        x5_up = self.attention_one(x4 + x5_up)
        # x5_up = x5_up + x4_a

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = self.attention_two(x3 + x6_up)
        # x6_up = x6_up + x3_a

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = self.attention_three(x2 + x7_up)
        # x7_up = x7_up + x2_a

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = self.attention_four(x1 + x8_up)
        # x8_up = x8_up + x1_a
        x9 = self.block_nine(x8_up)
        # x9 = F.dropout3d(x9, p=0.5, training=True)

        out = self.out_conv(x9)
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
#     writer = SummaryWriter('test')
#     model = VNet(n_channels=1, n_classes=2, normalization='instancenorm', has_dropout=False, activation=nn.ReLU)
#     tin = torch.zeros(1, 1, 128,160,208)
#     tout, cam = model(tin)
#     print(tout.size(), cam.size())
#     writer.add_graph(model, tin)
#     writer.close()
    