from torch.nn import functional as F
import torch.nn as nn
import torch

from functools import partial


def conv3d(in_channels, out_channels, kernel_size, bias, padding):
    return nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)

def create_conv(in_channels, out_channels, order, kernel_size, num_groups, padding):
    """
    Create a list of modules with together constitute a single conv layer with non-linearity
    and optional batchnorm/groupnorm.

    order (string) e.g. 1. 'gcr' -> groupnorm + conv + ReLU, 2. 'bcr' -> batchnorm + conv + ReLU

    Return:
        list of tuple (name, module)
    """

    modules = []
    for i, char in enumerate(order):
        if char == 'r': # Add ReLU layer
            modules.append(('ReLU', nn.ReLU(inplace=True)))
        elif char == 'e': # Add ELU layer
            modules.append(('ELU', nn.ELU(inplace=True)))
        elif char == 'c': # Add 3D Convlutional layer
            # add learnable bias only in the absence of batchnorm/groupnorm
            bias = not ('g' in order or 'b' in order)
            modules.append(('conv', conv3d(in_channels, out_channels, kernel_size, False, padding=padding)))
        elif char == 'g': # Add Group Norm layer
            before_conv = i < order.index('c')
            if before_conv: num_channels = in_channels
            else: num_channels = out_channels

            if num_channels < num_groups:
                num_groups = 1

            modules.append(('groupnorm', nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)))

        elif char == 'b': # Add Batch Norm Layer
            before_conv = i < order.index('c')
            if before_conv: modules.append(('batchnorm', nn.BatchNorm3d(in_channels)))
            else:           modules.append(('batchnorm', nn.BatchNorm3d(out_channels)))

    return modules

class SingleConv(nn.Sequential):
    """
    Basic convolutional module consisting of a Conv3d, non-linearity and optional batchnorm/groupnorm. The order
    of operations can be specified via the `order` parameter

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int or tuple): size of the convolving kernel
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple):
    """

    def __init__(self, in_channels, out_channels, order, kernel_size=3, num_groups=8, padding=1):
        super(SingleConv, self).__init__()

        for name, module in create_conv(in_channels, out_channels, order, kernel_size, num_groups, padding=padding):
            self.add_module(name, module)


class DoubleConv(nn.Sequential):
    """
    A module consisting of two consecutive convolution layers (e.g. BatchNorm3d+ReLU+Conv3d).
    We use (Conv3d+ReLU+GroupNorm3d) by default.
    Use padded convolutions to make sure that the output (H_out, W_out) is the same
    as (H_in, W_in), so that you don't have to crop in the decoder path.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        encoder (bool): if True we're in the encoder path, otherwise we're in the decoder
        kernel_size (int or tuple): size of the convolving kernel

        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding added to all three sides of the input
    """

    def __init__(self, in_channels, out_channels, encoder, order='gcr', kernel_size=3, num_groups=8, padding=1):
        super(DoubleConv, self).__init__()
        if encoder:
            # we're in the encoder path
            conv1_in_channels = in_channels
            conv1_out_channels = out_channels // 2

            if conv1_out_channels < in_channels:
                conv1_out_channels = in_channels
                
            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels
        else:
            # we're in the decoder path, decrease the number of channels in the 1st convolution
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels
            
        # conv1
        self.add_module('SingleConv1',
                        SingleConv(conv1_in_channels, conv1_out_channels, order, kernel_size, num_groups,
                                   padding=padding))
        # conv2
        self.add_module('SingleConv2',
                        SingleConv(conv2_in_channels, conv2_out_channels, order, kernel_size, num_groups,
                                   padding=padding))

class ResNetBlock(nn.Module):
    """
    Basic UNet block consisting of a SingleConv followed by the residual block.
    The SingleConv takes care of increasing/decreasing the number of channels and also ensures that the number
    of output channels is compatible with the residual block that follows.
    This block can be used instead of standard DoubleConv in the Encoder module.
    Motivated by: https://arxiv.org/abs/2006.14215

    Notice we use ELU instead of ReLU (order='cge') and put non-linearity after the groupnorm.
    """

    def __init__(self, in_channels, out_channels, encoder, kernel_size=3, order='cge', num_groups=8, **kwargs):
        super(ResNetBlock, self).__init__()

        # first convolution
        self.conv1 = SingleConv(in_channels, out_channels, kernel_size=kernel_size, order=order, num_groups=num_groups)
        
        # residual block
        self.conv2 = SingleConv(out_channels, out_channels, kernel_size=kernel_size, order=order, num_groups=num_groups)
       
        # remove non-linearity from the 3rd convolution since it's going to be applied after adding the residual
        n_order = order
        for c in 'rel': n_order = n_order.replace(c, '')

        self.conv3 = SingleConv(out_channels, out_channels, kernel_size=kernel_size, order=n_order,
                                num_groups=num_groups)

        # create non-linearity separately
        self.non_linearity = nn.ELU(inplace=True)

    def forward(self, x):
        # apply first convolution and save the output as a residual
        out = self.conv1(x)
        residual = out.copy()

        # residual block
        out = self.conv2(out)
        out = self.conv3(out)

        out += residual
        out = self.non_linearity(out)

        return out

class AbstractUpsampling(nn.Module):
    """
    Abstract class for upsampling. A given implementation should upsample a given 5D input tensor using either
    interpolation or learned transposed convolution.
    """

    def __init__(self, upsample):
        super(AbstractUpsampling, self).__init__()
        self.upsample = upsample

    def forward(self, encoder_features, x):
        # get the spatial dimensions of the output given the encoder_features
        output_size = encoder_features.size()[2:]
        # upsample the input and return
        return self.upsample(x, output_size)


class InterpolateUpsampling(AbstractUpsampling):
    """
    Args:
        mode (str): algorithm used for upsampling:
            'nearest' | 'linear' | 'bilinear' | 'trilinear' | 'area'. Default: 'nearest'
            used only if transposed_conv is False
    """

    def __init__(self, mode='nearest'):
        upsample = partial(self._interpolate, mode=mode)
        super().__init__(upsample)

    @staticmethod
    def _interpolate(x, size, mode):
        return F.interpolate(x, size=size, mode=mode)

class Encoder(nn.Module):
    """
    A single module from the encoder path consisting of the optional max
    pooling layer (one may specify the MaxPool kernel_size to be different
    than the standard (2,2,2), e.g. if the volumetric data is anisotropic
    (make sure to use complementary scale_factor in the decoder path) followed by
    a DoubleConv module.
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        conv_kernel_size (int or tuple): size of the convolving kernel
        apply_pooling (bool): if True use MaxPool3d before DoubleConv
        pool_kernel_size (int or tuple): the size of the window
        pool_type (str): pooling layer: 'max' or 'avg'
        basic_module(nn.Module): either ResNetBlock or DoubleConv
        conv_layer_order (string): determines the order of layers
            in `DoubleConv` module. See `DoubleConv` for more info.
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding added to all three sides of the input
    """

    def __init__(self, in_channels, out_channels, conv_kernel_size=3, apply_pooling=True,
                 pool_kernel_size=2, pool_type='max', basic_module='DoubleConv', 
                 num_groups=8, padding=1):
        super(Encoder, self).__init__()

        if apply_pooling:
            # Max Pooling is Default
            if pool_type == 'max':
                self.pooling = nn.MaxPool3d(kernel_size=pool_kernel_size)
            elif pool_type == 'avg':
                self.pooling = nn.AvgPool3d(kernel_size=pool_kernel_size)
        else: 
            self.pooling = None
              
        # Basic Module 
        if basic_module == 'DoubleConv':                                                                 # Encoder
            self.basic_module = DoubleConv(in_channels, out_channels, True,
                                            kernel_size=conv_kernel_size,
                                            num_groups=num_groups,
                                            padding=padding)
        elif basic_module == 'ResNetBlock':
            self.basic_module = ResNetBlock(in_channels, out_channels, True,
                                            kernel_size=conv_kernel_size,
                                            num_groups=num_groups,
                                            padding=padding)

    def forward(self, x):
        if self.pooling is not None:
            x = self.pooling(x)
        x = self.basic_module(x)
        return x

class Decoder(nn.Module):
    """
    A single module for decoder path consisting of the upsampling layer
    (either learned ConvTranspose3d or nearest neighbor interpolation) followed by a basic module (DoubleConv or ExtResNetBlock).
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        conv_kernel_size (int or tuple): size of the convolving kernel
        scale_factor (tuple): used as the multiplier for the image H/W/D in
            case of nn.Upsample or as stride in case of ConvTranspose3d, must reverse the MaxPool3d operation
            from the corresponding encoder
        basic_module(nn.Module): either ResNetBlock or DoubleConv
        conv_layer_order (string): determines the order of layers
            in `DoubleConv` module. See `DoubleConv` for more info.
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding added to all three sides of the input
        upsample (boole): should the input be upsampled
    """

    def __init__(self, in_channels, out_channels, conv_kernel_size=3, scale_factor=(2, 2, 2), num_groups=8, 
                 basic_module='DoubleConv', mode='nearest', padding=1, upsample=True):
        super(Decoder, self).__init__()

        # Interpolation Upsampling
        self.upsampling = InterpolateUpsampling(mode=mode)

        # concat joining
        self.joining = partial(self._joining, concat=True)                
            
        if basic_module == 'DoubleConv':
            self.basic_module = DoubleConv(in_channels, out_channels, False,
                                            kernel_size=conv_kernel_size,
                                            num_groups=num_groups,
                                            padding=padding)
        elif basic_module == 'ResNetBlock':
            self.basic_module = ResNetBlock(in_channels,out_channels, False,
                                            kernel_size = conv_kernel_size, num_groups=num_groups,
                                            padding=padding)

    def forward(self, encoder_features, x):
        x = self.upsampling(encoder_features=encoder_features, x=x)
        x = self.joining(encoder_features, x)
        x = self.basic_module(x)
        return x

    @staticmethod
    def _joining(encoder_features, x, concat):
        if concat:
            return torch.cat((encoder_features, x), dim=1)
        else:
            return encoder_features + x

# class InterpolateUpsampling(nn.Module):
#     """
#     Args:
#         mode (str): algorithm used for upsampling:
#             'nearest' | 'linear' | 'bilinear' | 'trilinear' | 'area'. Default: 'nearest'
#             used only if transposed_conv is False
#     """

#     def __init__(self, mode='nearest'):
#         # upsample = partial(self._interpolate, mode=mode)
#         # super().__init__(upsample)
#         self.mode = mode
#         super(InterpolateUpsampling, self).__init__()

#     def forward(self,size, x):
#         return F.interpolate(x, size=size, mode=self.mode)
