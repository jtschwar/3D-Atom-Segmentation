import models.utils as utils
import torch.nn as nn


class ResUNet3D(nn.Module):
    """
    Res3DUnet model from
    "Deep Residual 3D U-Net for Joint Segmentation" <https://arxiv.org/abs/2006.14215>`
    and "Residual 3D U-Net with Localization for Brain Tumor Segmentation" <https://link.springer.com/chapter/10.1007/978-3-031-08999-2_33>
    Uses `Residual Blocks` as a basic_module and nearest neighbor upsampling in the decoder 
    """

    def __init__(self, config, conv_kernel_size=3, pool_kernel_size=2,
                 conv_padding=1, **kwargs):
        super(ResUNet3D, self).__init__()

        f_maps = config['f_maps']; num_groups = config['num_groups']
        final_sigmoid = config['final_sigmoid']

        in_channels = 1; out_channels = 1; basic_module = 'ResNetBlock'

        # create encoder path
        self.encoders = []
        for i, out_feature_num in enumerate(f_maps):
            if i == 0:
                self.encoder = utils.Encoder(in_channels, out_feature_num,
                                apply_pooling=False,  # skip pooling in the first encoder
                                basic_module=basic_module,
                                conv_kernel_size=conv_kernel_size,
                                num_groups=num_groups,
                                padding=conv_padding)
            else:
                self.encoder = utils.Encoder(f_maps[i - 1], out_feature_num,
                                basic_module=basic_module,
                                conv_kernel_size=conv_kernel_size,
                                num_groups=num_groups,
                                pool_kernel_size=pool_kernel_size,
                                padding=conv_padding)
            self.encoders.append(self.encoder)
        self.encoders = nn.ModuleList(self.encoders)
       
        # create decoder path
        self.decoders = []
        reversed_f_maps = list(reversed(f_maps))
        for i in range(len(reversed_f_maps) - 1):
            # if basic_module == 'DoubleConv':
            #     in_feature_num = reversed_f_maps[i] + reversed_f_maps[i + 1]
            # else:
            #     in_feature_num = reversed_f_maps[i]
            in_feature_num = reversed_f_maps[i] + reversed_f_maps[i + 1]

            out_feature_num = reversed_f_maps[i + 1]

            decoder = utils.Decoder(in_feature_num, out_feature_num,
                            basic_module=basic_module,
                            conv_kernel_size=conv_kernel_size,
                            num_groups=num_groups,
                            padding=conv_padding)
            self.decoders.append(decoder)  
        self.decoders = nn.ModuleList(self.decoders)    
        
        # in the last layer a 1×1 convolution reduces the number of output channels to the number of labels
        self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)

        if final_sigmoid: self.final_activation = nn.Sigmoid()
        else:             self.final_activation = nn.Softmax(dim=1)

    def forward(self, x):

        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)

            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list (!!remember: it's the 1st in the list)
        encoders_features = encoders_features[1:]

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output of the previous decoder
            x = decoder(encoder_features, x)

        x = self.final_conv(x)

        # apply final_activation (i.e. Sigmoid or Softmax) only during prediction. During training the network outputs logits
        if not self.training and self.final_activation is not None:
            x = self.final_activation(x)

        return x