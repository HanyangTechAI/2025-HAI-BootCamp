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

    def _make_conv_layers():
        darknet = nn.Sequential(

        )
        return darknet

    def _make_fc_layers():
        fc = nn.Sequential(

        )
        return fc    

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)