from torch import nn


class Decoder(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            ngf=16,  # 64
            norm_layer=nn.LayerNorm,
            kernel_size_list=(3, 3, 5, 5, 5, 5),  # 5
            stride_list=(2, 2, 2, 2, 2, 2),  # 3
            padding_list=(1, 1, 2, 2, 2, 2),
            output_padding_list=(1, 1, 1, 1, 1, 1),
    ):
        super(Decoder, self).__init__()
        decoder = []
        assert len(kernel_size_list) == len(stride_list) == len(padding_list) == len(output_padding_list)
        self.image_sizes = []
        image_size = 1
        for k, s, p, op in zip(kernel_size_list, stride_list, padding_list, output_padding_list):
            image_size = (image_size - 1) * s - 2 * p + 1 * (k - 1) + op + 1
            self.image_sizes.append(image_size)
        n_upsampling = len(kernel_size_list)
        mult = 2 ** n_upsampling

        conv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=int(ngf * mult / 2),
            kernel_size=kernel_size_list[0],
            stride=stride_list[0],
            padding=padding_list[0],
            output_padding=output_padding_list[0]
        )
        decoder += [conv,
                    norm_layer([int(ngf * mult / 2), self.image_sizes[0], self.image_sizes[0]]),
                    nn.ReLU(True)]
        for i in range(1, n_upsampling):
            mult = 2 ** (n_upsampling - i)
            conv = nn.ConvTranspose2d(
                in_channels=ngf * mult,
                out_channels=int(ngf * mult / 2),
                kernel_size=kernel_size_list[i],
                stride=stride_list[i],
                padding=padding_list[i],
                output_padding=output_padding_list[i]
            )
            decoder += [conv,
                        norm_layer([int(ngf * mult / 2), self.image_sizes[i], self.image_sizes[i]]),
                        nn.ReLU(True)]
        decoder += [nn.Conv2d(ngf, out_channels, kernel_size=7, padding=7 // 2)]
        decoder += [nn.Sigmoid()]
        self.decode = nn.Sequential(*decoder)

    def forward(self, latte):
        return self.decode(latte)
