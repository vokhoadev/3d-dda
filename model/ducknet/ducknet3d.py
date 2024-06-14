import torch
from torch import nn
import pytorch_lightning as pl
from torch.nn import functional as F

class DuckNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1):
        super(DuckNetBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x

class DuckNet3D(pl.LightningModule):
    def __init__(self, img_height, img_width, img_depth, input_channels, out_classes, starting_filters):
        super(DuckNet3D, self).__init__()
        self.save_hyperparameters()

        self.input_layer = nn.Conv3d(input_channels, starting_filters, kernel_size=3, padding=1)

        self.encoder1 = DuckNetBlock(starting_filters, starting_filters * 2)
        self.encoder2 = DuckNetBlock(starting_filters * 2, starting_filters * 4)
        self.encoder3 = DuckNetBlock(starting_filters * 4, starting_filters * 8)
        self.encoder4 = DuckNetBlock(starting_filters * 8, starting_filters * 16)

        self.bottleneck = DuckNetBlock(starting_filters * 16, starting_filters * 32)

        self.up1 = nn.ConvTranspose3d(starting_filters * 32, starting_filters * 16, kernel_size=2, stride=2)
        self.decoder1 = DuckNetBlock(starting_filters * 32, starting_filters * 16)
        self.up2 = nn.ConvTranspose3d(starting_filters * 16, starting_filters * 8, kernel_size=2, stride=2)
        self.decoder2 = DuckNetBlock(starting_filters * 16, starting_filters * 8)
        self.up3 = nn.ConvTranspose3d(starting_filters * 8, starting_filters * 4, kernel_size=2, stride=2)
        self.decoder3 = DuckNetBlock(starting_filters * 8, starting_filters * 4)
        self.up4 = nn.ConvTranspose3d(starting_filters * 4, starting_filters * 2, kernel_size=2, stride=2)
        self.decoder4 = DuckNetBlock(starting_filters * 4, starting_filters * 2)

        self.output_layer = nn.Conv3d(starting_filters * 2, out_classes, kernel_size=1)

        # Choose loss function based on output classes
        if out_classes > 1:
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, x):
        x1 = self.input_layer(x)

        e1 = self.encoder1(F.max_pool3d(x1, 2))
        e2 = self.encoder2(F.max_pool3d(e1, 2))
        e3 = self.encoder3(F.max_pool3d(e2, 2))
        e4 = self.encoder4(F.max_pool3d(e3, 2))

        b = self.bottleneck(F.max_pool3d(e4, 2))

        d1 = self.up1(b)
        d1 = torch.cat((d1, e4), dim=1)
        d1 = self.decoder1(d1)

        d2 = self.up2(d1)
        d2 = torch.cat((d2, e3), dim=1)
        d2 = self.decoder2(d2)

        d3 = self.up3(d2)
        d3 = torch.cat((d3, e2), dim=1)
        d3 = self.decoder3(d3)

        d4 = self.up4(d3)
        d4 = torch.cat((d4, e1), dim=1)
        d4 = self.decoder4(d4)

        output = self.output_layer(d4)
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        if self.hparams.out_classes > 1:
            # For multi-class segmentation
            y = y.squeeze(1)  # Remove channel dimension for CrossEntropyLoss
            loss = self.loss_fn(y_hat, y)
        else:
            # For binary segmentation
            loss = self.loss_fn(y_hat, y.float())  # Ensure y is float for BCEWithLogitsLoss
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        if self.hparams.out_classes > 1:
            # For multi-class segmentation
            y = y.squeeze(1)
            loss = self.loss_fn(y_hat, y)
        else:
            # For binary segmentation
            loss = self.loss_fn(y_hat, y.float())
        self.log('val_loss', loss)
        return loss
