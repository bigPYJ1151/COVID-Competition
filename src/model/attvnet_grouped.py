import torch
from torch import nn
import torch.nn.functional as F
# from torch.utils.tensorboard import SummaryWriter

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

    class SAttention(nn.Module):
        
        def __init__(self, inplane, ratio=4, groups=1):
            super().__init__()

            self.inplane = inplane
            self.groups = groups
            self.ratio = ratio
            self.groups_per = inplane // groups

            self.conv1 = nn.Conv3d(inplane, inplane // ratio, kernel_size=1, groups=groups)
            self.conv2 = nn.Conv3d(inplane, inplane // ratio, kernel_size=1, groups=groups)
            self.conv3 = nn.Conv3d(inplane, inplane, kernel_size=1, groups=groups)

            self.softmax = nn.Softmax(-1)

        def forward(self, x):
            b, c, d, h, w = x.size()
            
            query = self.conv1(x).view(b * self.groups, self.groups_per // self.ratio, d * w * h).permute(0, 2, 1)
            key = self.conv2(x).view(b * self.groups, self.groups_per // self.ratio, d * w * h)
            
            corr_map = torch.bmm(query, key) 
            attention_map = self.softmax(corr_map)
            value = self.conv3(x).view(b * self.groups, self.groups_per, d * w * h)

            ans = torch.bmm(value, attention_map.permute(0, 2, 1))
            
            return ans.view(b, c, d, h, w)


    class CAttention(nn.Module):
        
        def __init__(self, inplane, groups=1):
            super().__init__()

            self.inplane = inplane
            self.groups = groups
            self.groups_per = inplane // groups

            # self.conv1 = nn.Conv3d(inplane, inplane, kernel_size=3, groups=groups, stride=1, padding=1)
            # self.conv2 = nn.Conv3d(inplane, inplane, kernel_size=3, groups=groups, stride=1, padding=1)
            # self.conv3 = nn.Conv3d(inplane, inplane, kernel_size=3, groups=groups, stride=1, padding=1)

            self.softmax = nn.Softmax(-1)

        def forward(self, x):
            b, c, d, h, w = x.size()

            # query = self.conv1(x).view(b * self.groups, self.groups_per, d * h * w)
            # key = self.conv2(x).view(b * self.groups, self.groups_per, d * h * w).permute(0, 2, 1)

            query = x.view(b * self.groups, self.groups_per, d * h * w)
            key = x.view(b * self.groups, self.groups_per, d * h * w).permute(0, 2, 1)

            corr_map = torch.bmm(query, key) 
            attention_map = self.softmax(corr_map)
            # value = self.conv3(x).view(b * self.groups, self.groups_per, d * h * w)
            value = x.view(b * self.groups, self.groups_per, d * h * w)

            ans = torch.bmm(attention_map, value)

            return ans.view(b, c, d, h, w)


    def __init__(self, inplane, normalization, activation, ratio=4, groups=1):
        super().__init__()
        
        self.i_conv = ConvBlock(1, inplane, inplane, normalization=normalization, activation=activation) 
        self.spatial_attention = SCAttention.SAttention(inplane, ratio, groups)
        self.channel_attention = SCAttention.CAttention(inplane, groups)
        self.o_conv = ConvBlock(1, inplane, inplane, normalization=normalization, activation=activation) 

    def forward(self, x):
        x = self.i_conv(x)
        sa = self.spatial_attention(x)
        ca = self.channel_attention(x)

        return self.o_conv(x + sa + ca)

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
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False, activation=nn.ReLU, ratio=4, groups=1):
        super(VNet, self).__init__()
        self.has_dropout = has_dropout

        self.block_one = ConvBlock(1, n_channels, n_filters, normalization=normalization, activation=activation)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization, activation=activation)

        self.block_two = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization, activation=activation)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization, activation=activation)

        self.block_three = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization, activation=activation)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization, activation=activation)

        self.block_four = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization, activation=activation)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization, activation=activation)

        # self.attention_one = SCAttention(n_filters * 16, normalization, activation, ratio=ratio, groups=groups)

        self.block_five = ConvBlock(3, n_filters * 16, n_filters * 16, normalization=normalization, activation=activation)
        self.block_five_up = UpsamplingDeconvBlock(n_filters * 16, n_filters * 8, normalization=normalization, activation=activation)

        self.block_six = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization, activation=activation)
        self.block_six_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4, normalization=normalization, activation=activation)
        # self.attention_two = SCAttention(n_filters * 4, normalization, activation, ratio=ratio, groups=groups)

        # self.helper_out2 = nn.Conv3d(n_filters * 4, n_classes, 1, padding=0) 

        self.block_seven = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization, activation=activation)
        self.block_seven_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, normalization=normalization, activation=activation)
        # self.attention_three = SCAttention(n_filters * 2, normalization, activation, ratio=4, groups=1)

        # self.helper_out3 = nn.Conv3d(n_filters * 2, n_classes, 1, padding=0) 

        self.block_eight = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization, activation=activation)
        self.block_eight_up = UpsamplingDeconvBlock(n_filters * 2, n_filters, normalization=normalization, activation=activation)
        # self.attention_four = SCAttention(n_filters, normalization, activation, ratio=4, groups=1)

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

        # x4_dw = self.attention_one(x4_dw)

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
        # x5_up = self.attention_one(x4 + x5_up)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        # x6_up = self.attention_two(x3 + x6_up)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        # x7_up = self.attention_three(x2 + x7_up)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        # x8_up = self.attention_four(x1 + x8_up)
        x8_up = x8_up + x1
        x9 = self.block_nine(x8_up)
        # x9 = F.dropout3d(x9, p=0.5, training=True)

        out = self.out_conv(x9)
        
        # return [out, self.helper_out3(x7_up), self.helper_out2(x6_up)]
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
#     writer = SummaryWriter('tensorboard')
#     model = VNet(n_channels=1, n_classes=3, normalization='instancenorm', has_dropout=False, activation=nn.ReLU)
#     tin = torch.zeros(1, 1, 128, 128, 128)
#     print(model(tin).size())
#     writer.add_graph(model, tin)
#     writer.close()
    