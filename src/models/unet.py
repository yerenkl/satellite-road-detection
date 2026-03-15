import torch
import torch.nn as nn
import torchvision.models as models


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class ResNet34_UNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)

        # ----- Encoder (ResNet34) -----
        self.initial = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu
        )                      # (64, H/2, W/2)

        self.maxpool = resnet.maxpool

        self.encoder1 = resnet.layer1  # 64
        self.encoder2 = resnet.layer2  # 128
        self.encoder3 = resnet.layer3  # 256
        self.encoder4 = resnet.layer4  # 512

        # ----- Bottleneck -----
        self.bottleneck = ConvBlock(512, 1024)

        # ----- Decoder -----
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = ConvBlock(768, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = ConvBlock(384, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = ConvBlock(192, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = ConvBlock(128, 64)

        self.up0 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.dec0 = ConvBlock(64, 64)

        # ----- Output -----
        self.out = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):

        # Encoder
        x0 = self.initial(x)      # 64
        x1 = self.encoder1(self.maxpool(x0))  # 64
        x2 = self.encoder2(x1)    # 128
        x3 = self.encoder3(x2)    # 256
        x4 = self.encoder4(x3)    # 512

        # Bottleneck
        x = self.bottleneck(x4)

        # Decoder
        x = self.up4(x)
        x = torch.cat([x, x3], dim=1)
        x = self.dec4(x)

        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.dec3(x)

        x = self.up2(x)
        x = torch.cat([x, x1], dim=1)
        x = self.dec2(x)

        x = self.up1(x)
        x = torch.cat([x, x0], dim=1)
        x = self.dec1(x)

        x = self.up0(x)
        x = self.dec0(x)

        return self.out(x)


if __name__ == "__main__":
    model = ResNet34_UNet(pretrained=True)

    x = torch.randn(1, 3, 256, 256)
    y = model(x)

    print(y.shape)  # should be (1, 1, 256, 256)