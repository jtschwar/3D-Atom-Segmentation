from utils import number_of_features_per_level, get_class
import models.utils as utils
import torch.nn as nn


class UNet3D(nn.Module):
    """
    3DUnet model from
    `"3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"
        <https://arxiv.org/pdf/1606.06650.pdf>`.
    Uses `DoubleConv` as a basic_module and nearest neighbor upsampling in the decoder 
    """

    def __init__(self, config, conv_kernel_size=3, pool_kernel_size=2,
                 conv_padding=1, **kwargs):
        super(UNet3D, self).__init__()
        
        in_channels = config['in_channels']; out_channels = config['out_channels']
        f_maps = config['f_maps']; num_groups = config['num_groups']
        final_sigmoid = config['final_sigmoid']; basic_module = config['basic_module']

        self.encoders = []
        for i, out_feature_num in enumerate(f_maps):
            if i == 0:
                encoder = utils.Encoder(in_channels, out_feature_num,
                                apply_pooling=False,  # skip pooling in the firs encoder
                                basic_module=basic_module,
                                conv_kernel_size=conv_kernel_size,
                                num_groups=num_groups,
                                padding=conv_padding)
            else:
                encoder = utils.Encoder(f_maps[i - 1], out_feature_num,
                                basic_module=basic_module,
                                conv_kernel_size=conv_kernel_size,
                                num_groups=num_groups,
                                pool_kernel_size=pool_kernel_size,
                                padding=conv_padding)
            self.encoders.append(encoder)

        self.decoders = []
        reversed_f_maps = list(reversed(f_maps))
        for i in range(len(reversed_f_maps) - 1):
            if basic_module == basic_module:
                in_feature_num = reversed_f_maps[i] + reversed_f_maps[i + 1]
            else:
                in_feature_num = reversed_f_maps[i]

            out_feature_num = reversed_f_maps[i + 1]

            decoder = utils.Decoder(in_feature_num, out_feature_num,
                            basic_module=basic_module,
                            conv_kernel_size=conv_kernel_size,
                            num_groups=num_groups,
                            padding=conv_padding)
            self.decoders.append(decoder)    

        # in the last layer a 1×1 convolution reduces the number of output
        # channels to the number of labels
        self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)

        if final_sigmoid:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Softmax(dim=1)

    def forward(self, x):
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)

        x = self.final_conv(x)

        # apply final_activation (i.e. Sigmoid or Softmax) only during prediction. During training the network outputs logits
        if not self.training and self.final_activation is not None:
            x = self.final_activation(x)

        return x