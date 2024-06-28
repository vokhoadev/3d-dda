from scipy import interpolate
import torch
import torch.nn as nn
import pytorch_lightning as pl

kernel_initializer = 'he_uniform'
interpolation = "linear"

class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, block_type, repeat=1):
        super(ConvBlock3D, self).__init__()
        self.blocks = nn.Sequential(
            *[self._make_block(in_channels, out_channels, block_type) for _ in range(repeat)]
        )

    def _make_block(self, in_channels, out_channels, block_type):
        if block_type == 'duckv2':
            return nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True)
            )
        elif block_type == 'resnet':
            return nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        return self.blocks(x)

class DuckNet3D(pl.LightningModule):
    def __init__(self, in_channels, out_channels, starting_filters):
        super(DuckNet3D, self).__init__()
        self.starting_filters = starting_filters

        self.conv1 = nn.Conv3d(in_channels, starting_filters * 2, kernel_size=2, stride=2, padding=1)
        self.conv2 = nn.Conv3d(starting_filters * 2, starting_filters * 4, kernel_size=2, stride=2, padding=1)
        self.conv3 = nn.Conv3d(starting_filters * 4, starting_filters * 8, kernel_size=2, stride=2, padding=1)
        self.conv4 = nn.Conv3d(starting_filters * 8, starting_filters * 16, kernel_size=2, stride=2, padding=1)
        self.conv5 = nn.Conv3d(starting_filters * 16, starting_filters * 32, kernel_size=2, stride=2, padding=1)

        self.block1 = ConvBlock3D(in_channels, starting_filters, 'duckv2', repeat=1)
        self.block2 = ConvBlock3D(starting_filters * 2, starting_filters * 2, 'duckv2', repeat=1)
        self.block3 = ConvBlock3D(starting_filters * 4, starting_filters * 4, 'duckv2', repeat=1)
        self.block4 = ConvBlock3D(starting_filters * 8, starting_filters * 8, 'duckv2', repeat=1)
        self.block5 = ConvBlock3D(starting_filters * 16, starting_filters * 16, 'duckv2', repeat=1)

        self.block51 = ConvBlock3D(starting_filters * 32, starting_filters * 32, 'resnet', repeat=2)
        self.block53 = ConvBlock3D(starting_filters * 32, starting_filters * 16, 'resnet', repeat=2)

        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.final_conv = nn.Conv3d(starting_filters, out_channels, kernel_size=1)


    def forward(self, x):
        p1 = self.conv1(x)
        p2 = self.conv2(p1)
        p3 = self.conv3(p2)
        p4 = self.conv4(p3)
        p5 = self.conv5(p4)

        t0 = self.block1(x)

        l1i = self.conv1(t0)
        s1 = p1 + l1i
        t1 = self.block2(s1)

        l2i = self.conv2(t1)
        s2 = p2 + l2i
        t2 = self.block3(s2)

        l3i = self.conv3(t2)
        s3 = p3 + l3i
        t3 = self.block4(s3)

        l4i = self.conv4(t3)
        s4 = p4 + l4i
        t4 = self.block5(s4)

        l5i = self.conv5(t4)
        s5 = p5 + l5i
        t51 = self.block51(s5)
        t53 = self.block53(t51)

        l5o = self.upsample(t53)
        c4 = t4 + l5o
        q4 = self.block4(c4)

        l4o = self.upsample(q4)
        c3 = t3 + l4o
        q3 = self.block3(c3)

        l3o = self.upsample(q3)
        c2 = t2 + l3o
        q6 = self.block2(c2)

        l2o = self.upsample(q6)
        c1 = t1 + l2o
        q1 = self.block1(c1)

        l1o = self.upsample(q1)
        c0 = t0 + l1o
        z1 = self.block1(c0)

        output = self.final_conv(z1)
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        # Ensure y has the correct shape
        if y.shape[1] != self.hparams.out_channels:
            raise AssertionError("labels should have a channel with length equal to the number of output channels.")

        # Resize labels to match the output if necessary
        if y.shape[2:] != y_hat.shape[2:]:
            y = interpolate(y, size=y_hat.shape[2:], mode='trilinear', align_corners=True)

        dice_loss = self.dice_loss(y_hat, y)
        ce_loss = self.ce_loss(y_hat, y.argmax(dim=1))  # CrossEntropyLoss expects class indices
        loss = dice_loss + ce_loss

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        # Ensure y has the correct shape
        if y.shape[1] != self.hparams.out_channels:
            raise AssertionError("labels should have a channel with length equal to the number of output channels.")

        # Resize labels to match the output if necessary
        if y.shape[2:] != y_hat.shape[2:]:
            y = interpolate(y, size=y_hat.shape[2:], mode='trilinear', align_corners=True)

        dice_loss = self.dice_loss(y_hat, y)
        ce_loss = self.ce_loss(y_hat, y.argmax(dim=1))
        loss = dice_loss + ce_loss

        self.log('val_loss', loss)
        return loss