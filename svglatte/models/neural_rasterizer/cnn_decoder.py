from torch import nn

norm_layer_name_dict = {
    "layernorm": lambda c, h, w: nn.LayerNorm([c, h, w]),
    "batchnorm": lambda c, h, w: nn.BatchNorm2d(c),
}


class Decoder(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            norm_layer_name,
            # residual_connections=False,
            n_filters_in_last_conv_layer,  # 64
            kernel_size_list=(3, 3, 5, 5, 5, 5),  # 5
            stride_list=(2, 2, 2, 2, 2, 2),  # 3
            padding_list=(1, 1, 2, 2, 2, 2),
            output_padding_list=(1, 1, 1, 1, 1, 1),
    ):
        assert len(kernel_size_list) == len(stride_list) == len(padding_list) == len(output_padding_list)
        super(Decoder, self).__init__()

        self.image_sizes = []

        image_size = 1
        for k, s, p, op in zip(kernel_size_list, stride_list, padding_list, output_padding_list):
            image_size = (image_size - 1) * s - 2 * p + 1 * (k - 1) + op + 1
            self.image_sizes.append(image_size)
        n_upsampling = len(kernel_size_list)
        mult = 2 ** n_upsampling

        norm_layer = norm_layer_name_dict[norm_layer_name]

        conv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=int(n_filters_in_last_conv_layer * mult / 2),
            kernel_size=kernel_size_list[0],
            stride=stride_list[0],
            padding=padding_list[0],
            output_padding=output_padding_list[0]
        )
        decoder = []
        decoder += [conv,
                    norm_layer(
                        c=int(n_filters_in_last_conv_layer * mult / 2),
                        h=self.image_sizes[0],
                        w=self.image_sizes[0],
                    ),
                    nn.ReLU(True)]
        for i in range(1, n_upsampling):
            mult = 2 ** (n_upsampling - i)
            conv = nn.ConvTranspose2d(
                in_channels=n_filters_in_last_conv_layer * mult,
                out_channels=int(n_filters_in_last_conv_layer * mult / 2),
                kernel_size=kernel_size_list[i],
                stride=stride_list[i],
                padding=padding_list[i],
                output_padding=output_padding_list[i]
            )
            decoder += [conv,
                        norm_layer(
                            c=int(n_filters_in_last_conv_layer * mult / 2),
                            h=self.image_sizes[i],
                            w=self.image_sizes[i],
                        ),
                        nn.ReLU(True)]
        decoder += [nn.Conv2d(n_filters_in_last_conv_layer, out_channels, kernel_size=7, padding=7 // 2)]
        decoder += [nn.Sigmoid()]
        self.decode = nn.Sequential(*decoder)

    def forward(self, latte):
        return self.decode(latte)
