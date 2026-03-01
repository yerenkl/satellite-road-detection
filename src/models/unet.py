from torch import nn
import torch

class Unet(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        
        self.encoder = nn.ModuleList()

        self.encoder = nn.ModuleList([
            self.conv_block(3, 64),
            self.conv_block(64, 128),
            self.conv_block(128, 256),
            self.conv_block(256, 512)
        ])

        self.bottleneck= nn.Sequential(nn.Conv2d(512, 1024, 3, 1, 1),
                    nn.ReLU(),
                    nn.Conv2d(1024, 1024, 3, 1, 1),
                    nn.ReLU())
        
        self.maxpooling = nn.MaxPool2d(2, 2)
        
        self.upsample = nn.ModuleList()
        self.decoder = nn.ModuleList()

        upsample_channels = [1024, 512, 256, 128]
        decoder_channels = [512, 256, 128, 64]
        self.upsample = nn.ModuleList([
            nn.ConvTranspose2d(in_ch, out_ch, 2, 2)
            for in_ch, out_ch in zip(upsample_channels, decoder_channels)
        ])

        self.decoder = nn.ModuleList([
            self.conv_block(1024, 512),
            self.conv_block(512, 256),
            self.conv_block(256, 128),
            self.conv_block(128, 64, n_convs=3)
        ])
        
    def conv_block(self, in_ch, out_ch, n_convs=2):
        layers = []
        for i in range(n_convs):
            layers.append(nn.Conv2d(in_ch if i == 0 else out_ch, out_ch, 3, 1, 1))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        x_s = []
        for layer in self.encoder:
            x = layer(x)
            x_s.append(x)
            x = self.maxpooling(x)

        x = self.bottleneck(x)

        for up, dec, skip in zip(self.upsample, self.decoder, reversed(x_s)):
            x = up(x)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)
        return x 
    
        
if __name__ == "__main__":
    model = Unet(1)
    x = torch.randn(1, 3, 256, 256)
    out = model(x)
    print(out.shape)
        