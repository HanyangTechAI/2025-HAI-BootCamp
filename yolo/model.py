import torch
import torch.nn as nn

S = 7
B = 5
C = 20

class YOLO(nn.Module):
    def __init__(self, init_weight=True):
        super(YOLO, self).__init__()

        if init_weight:
            self._initialize_weights()

        self.darknet = self._make_conv_layers()
        self.fully_connected_layer = self._make_fc_layers()

    def forward(self, x):
        x = self.darknet(x)
        x = self.fully_connected_layer(x)
        return x

    def _make_conv_layers(self):
        darknet = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LeakyReLU(0.1),

            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LeakyReLU(0.1),

            nn.Conv2d(in_channels=192, out_channels=128, kernel_size=1, padding=0),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, padding=0),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LeakyReLU(0.1),

            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, padding=0),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1), 
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, padding=0),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, padding=0),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1), 
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, padding=0),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, padding=0),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),  
            nn.LeakyReLU(0.1),

            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, padding=0),
            nn.LeakyReLU(0.1),   
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, padding=0),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1),

            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),   
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),

        )
        return darknet