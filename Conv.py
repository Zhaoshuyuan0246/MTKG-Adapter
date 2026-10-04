import torch
import torch.nn as nn
import math

class PretrainedConv1D(nn.Module):
    def __init__(self, pretrained_model):
        """
        Wrap the convolutional part + fully-connected mapping of ConvTransE,
        producing [bsz, h_dim].

        Args:
            pretrained_model (nn.Module): model containing ``decoder_ob``.
        """
        super(PretrainedConv1D, self).__init__()

        # Step 1: extract the convolution layer
        pretrained_conv = pretrained_model.decoder_ob.conv1
        weight_conv = pretrained_conv.weight.data
        bias_conv = pretrained_conv.bias.data

        self.conv_layer = nn.Conv1d(
            in_channels=pretrained_conv.in_channels,
            out_channels=pretrained_conv.out_channels,
            kernel_size=pretrained_conv.kernel_size[0],
            stride=pretrained_conv.stride[0],
            padding=pretrained_conv.padding[0]
        )

        self.conv_layer.weight.data.copy_(weight_conv)
        self.conv_layer.bias.data.copy_(bias_conv)
        for param in self.conv_layer.parameters():
            param.requires_grad = False

        # Step 2: extract BatchNorm and dropout
        self.bn1 = pretrained_model.decoder_ob.bn1
        self.feature_map_drop = pretrained_model.decoder_ob.feature_map_drop

        # Step 3: extract the fully-connected layer
        pretrained_fc = pretrained_model.decoder_ob.fc
        self.fc = nn.Linear(
            in_features=pretrained_fc.in_features,
            out_features=pretrained_fc.out_features
        )
        self.fc.weight.data.copy_(pretrained_fc.weight.data)
        self.fc.bias.data.copy_(pretrained_fc.bias.data)
        for param in self.fc.parameters():
            param.requires_grad = False

        # Step 4: extract the BatchNorm after the FC
        self.bn2 = pretrained_model.decoder_ob.bn2

    def forward(self, x):
        """
        Forward pass:
            - Conv1d -> BatchNorm -> ReLU -> Dropout
            - Flatten -> Linear -> BatchNorm -> ReLU

        Input shape: [bsz, 2, h_dim]
        Output shape: [bsz, h_dim]
        """
        # Conv + BN + ReLU + Dropout
        x = self.conv_layer(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.feature_map_drop(x)

        # Flatten
        bsz, c, h_dim = x.shape
        x = x.view(bsz, -1)  # shape: [bsz, c*h_dim]

        # Fully Connected
        x = self.fc(x)
        x = self.bn2(x)
        x = torch.relu(x)

        return x