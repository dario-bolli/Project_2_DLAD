import torch
import torch.nn.functional as F
import torchvision.models.resnet as resnet


class BasicBlockWithDilation(torch.nn.Module):
    """Workaround for prohibited dilation in BasicBlock in 0.4.0"""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlockWithDilation, self).__init__()
        if norm_layer is None:
            norm_layer = torch.nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        self.conv1 = resnet.conv3x3(inplanes, planes, stride=stride)
        self.bn1 = norm_layer(planes)
        self.relu = torch.nn.ReLU()
        self.conv2 = resnet.conv3x3(planes, planes, dilation=dilation)
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


_basic_block_layers = {
    'resnet18': (2, 2, 2, 2),
    'resnet34': (3, 4, 6, 3),
}


def get_encoder_channel_counts(encoder_name):
    is_basic_block = encoder_name in _basic_block_layers
    ch_out_encoder_bottleneck = 512 if is_basic_block else 2048
    ch_out_encoder_4x = 64 if is_basic_block else 256
    return ch_out_encoder_bottleneck, ch_out_encoder_4x


class Encoder(torch.nn.Module):
    def __init__(self, name, **encoder_kwargs):
        super().__init__()
        encoder = self._create(name, **encoder_kwargs)
        del encoder.avgpool
        del encoder.fc
        self.encoder = encoder

    def _create(self, name, **encoder_kwargs):
        if name not in _basic_block_layers.keys():
            fn_name = getattr(resnet, name)
            model = fn_name(**encoder_kwargs)
        else:
            # special case due to prohibited dilation in the original BasicBlock
            pretrained = encoder_kwargs.pop('pretrained', False)
            progress = encoder_kwargs.pop('progress', True)
            model = resnet._resnet(
                name, BasicBlockWithDilation, _basic_block_layers[name], pretrained, progress, **encoder_kwargs
            )
        replace_stride_with_dilation = encoder_kwargs.get('replace_stride_with_dilation', (False, False, False))
        assert len(replace_stride_with_dilation) == 3
        if replace_stride_with_dilation[0]:
            model.layer2[0].conv2.padding = (2, 2)
            model.layer2[0].conv2.dilation = (2, 2)
        if replace_stride_with_dilation[1]:
            model.layer3[0].conv2.padding = (2, 2)
            model.layer3[0].conv2.dilation = (2, 2)
        if replace_stride_with_dilation[2]:
            model.layer4[0].conv2.padding = (2, 2)
            model.layer4[0].conv2.dilation = (2, 2)
        return model

    def update_skip_dict(self, skips, x, sz_in):
        rem, scale = sz_in % x.shape[3], sz_in // x.shape[3]
        assert rem == 0
        skips[scale] = x

    def forward(self, x):
        """
        DeepLabV3+ style encoder
        :param x: RGB input of reference scale (1x)
        :return: dict(int->Tensor) feature pyramid mapping downscale factor to a tensor of features
        """
        out = {1: x}
        sz_in = x.shape[3]

        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        self.update_skip_dict(out, x, sz_in)

        x = self.encoder.maxpool(x)
        self.update_skip_dict(out, x, sz_in)

        x = self.encoder.layer1(x)
        self.update_skip_dict(out, x, sz_in)

        x = self.encoder.layer2(x)
        self.update_skip_dict(out, x, sz_in)

        x = self.encoder.layer3(x)
        self.update_skip_dict(out, x, sz_in)

        x = self.encoder.layer4(x)
        self.update_skip_dict(out, x, sz_in)

        return out


class DecoderDeeplabV3p(torch.nn.Module):
    def __init__(self, bottleneck_ch, skip_4x_ch, num_out_ch):
        super(DecoderDeeplabV3p, self).__init__()

        # TODO: Implement a proper decoder with skip connections instead of the following
        #48 is the best channels number according to paper
        self.features_to_concatenation = torch.nn.Sequential(torch.nn.Conv2d(skip_4x_ch, 48, kernel_size=1, stride=1),
                                                            torch.nn.BatchNorm2d(48),torch.nn.ReLU())
        self.concatenation_to_predictions = torch.nn.Sequential(torch.nn.Conv2d(bottleneck_ch+48, num_out_ch, kernel_size=3, padding=2),
                                                                torch.nn.BatchNorm2d(num_out_ch),
                                                                torch.nn.ReLU(),        
                                                                torch.nn.Conv2d(num_out_ch, num_out_ch, kernel_size=3, padding=2))

    def forward(self, features_bottleneck, features_skip_4x):
        """
        DeepLabV3+ style decoder
        :param features_bottleneck: bottleneck features of scale > 4
        :param features_skip_4x: features of encoder of scale == 4
        :return: features with 256 channels and the final tensor of predictions
        """
        # TODO: Implement a proper decoder with skip connections instead of the following; keep returned
        #       tensors in the same order and of the same shape.
        #upsample ASPP features by 4
        features_ASPP = F.interpolate(
            features_bottleneck, size=features_skip_4x.shape[2:], mode='bilinear', align_corners=False
        )
        #1x1 conv2d on lowest feature (skip)
        features_skip = self.features_to_concatenation(features_skip_4x)
        #concatenation of lowest feature and upsampled output of ASPP
        features_cat = torch.cat([features_skip,features_ASPP], dim=1)
        #2 3x3 conv2d on concatenated features
        predictions = self.concatenation_to_predictions(features_cat)
        return predictions, features_cat

class Decoder(torch.nn.Module):
    def __init__(self, attention_ch, out_DeepLab_ch, num_out_ch):
        super(DecoderDeeplabV3p, self).__init__()

        # TODO: Implement a proper decoder with skip connections instead of the following
        #48 is the best channels number according to paper
        self.concatenation_to_predictions = torch.nn.Sequential(torch.nn.Conv2d(attention_ch+out_DeepLab_ch, num_out_ch, kernel_size=3, padding=2),
                                                                torch.nn.BatchNorm2d(num_out_ch),
                                                                torch.nn.ReLU(),        
                                                                torch.nn.Conv2d(num_out_ch, num_out_ch, kernel_size=3, padding=2),
                                                                torch.nn.BatchNorm2d(num_out_ch))

    def forward(self, features_attention, features_out_decoder):
        """
        DeepLabV3+ style decoder
        :param features_bottleneck: bottleneck features of scale > 4
        :param features_skip_4x: features of encoder of scale == 4
        :return: features with 256 channels and the final tensor of predictions
        """
        # TODO: Implement a proper decoder with skip connections instead of the following; keep returned
        #       tensors in the same order and of the same shape.

        #concatenation of features out of self-attention module and out of decoderDeeplab module
        features_cat = torch.cat([features_attention,features_out_decoder], dim=1)
        #2 3x3 conv2d on concatenated features
        predictions = self.concatenation_to_predictions(features_cat)
        return predictions, features_cat

class ASPPpart(torch.nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation):
        super().__init__(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
        )

class ASPP(torch.nn.Module):
    def __init__(self, in_channels, out_channels, rates=(3, 6, 9)):
        super().__init__()
        # TODO
        modules = []
        rates = [2*x for x in rates]
        modules.append(ASPPpart(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1))
        for rate in rates:
            modules.append(ASPPpart(in_channels, out_channels, kernel_size=3, stride=1, padding=rate, dilation=rate))
       
        #global_avg = torch.nn.AdaptiveAvgPool2d(1) does not work gives [256,512, H, W] instead of [256,256, H,W]
        # therefore apply convolution with correct output channels
        global_avg = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1),
                                         torch.nn.Conv2d(in_channels, out_channels, kernel_size = 1),
                                         torch.nn.BatchNorm2d(out_channels),
                                         torch.nn.ReLU())
        modules.append(global_avg)
        self.aspp_convs = torch.nn.ModuleList(modules)
        # At this stage when called, already concatenated so we know how many out channels we have for each conv.
        # So total after concatenation of all diff layers of conv is len(self.aspp_convs)*out_channels
        self.conv_1x1 = torch.nn.Sequential(torch.nn.Conv2d(out_channels*len(self.aspp_convs), out_channels, kernel_size = 1),
                                            torch.nn.BatchNorm2d(out_channels),
                                            torch.nn.ReLU())
        #self.conv_out = ASPPpart(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1)

    def forward(self, x):
        # TODO: Implement ASPP properly instead of the following
        res = []                           
        features_h = x.size()[2] # is height of feature map image
        features_w = x.size()[3] # is width of feature map image
        for layer in self.aspp_convs:
            res.append(layer(x))
        #res[4] is the output of the average pooling but has h= 1, w = 1 so we upsample it to the needed height and width
        res[4] = F.interpolate(res[4],(features_h, features_w), mode = 'bilinear', align_corners=False)
        res = torch.cat(res, dim = 1)
        return self.conv_1x1(res)
        #out = self.conv_out(x)
        #return out



class SelfAttention(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.attention = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        with torch.no_grad():
            self.attention.weight.copy_(torch.zeros_like(self.attention.weight))

    def forward(self, x):
        features = self.conv(x)
        attention_mask = torch.sigmoid(self.attention(x))
        return features * attention_mask


class SqueezeAndExcitation(torch.nn.Module):
    """
    Squeeze and excitation module as explained in https://arxiv.org/pdf/1709.01507.pdf
    """
    def __init__(self, channels, r=16):
        super(SqueezeAndExcitation, self).__init__()
        self.transform = torch.nn.Sequential(
            torch.nn.Linear(channels, channels // r),
            torch.nn.ReLU(),
            torch.nn.Linear(channels // r, channels),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        N, C, H, W = x.shape
        squeezed = torch.mean(x, dim=(2, 3)).reshape(N, C)
        squeezed = self.transform(squeezed).reshape(N, C, 1, 1)
        return x * squeezed
