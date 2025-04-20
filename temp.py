import torch
import torch.nn as nn


class YourModel(nn.Module):
    def __init__(self):
        super(YourModel, self).__init__()
        # Assuming you have defined your layers in __init__
        self.conv3 = nn.Conv2d(128, 160, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        # Other layers in your network

    def forward(self, x):
        # Your forward pass implementation
        out1 = x[0]  # Output from the first stage
        out2 = x[1]  # Output from the second stage
        out3 = x[2]  # Output from the third stage
        out4 = x[3]  # Output from the fourth stage

        # Perform convolutions to adjust channel sizes and stride
        out3 = self.conv3(out3)
        out4 = self.conv4(out4)

        # Upsample to match desired size
        out3 = self.upsample(out3)

        return out1, out2, out3, out4


# Create an instance of your model
model = YourModel()

# Assuming your outputs are stored in a list
outputs = [torch.randn(1, 32, 160, 160),
           torch.randn(1, 64, 80, 80),
           torch.randn(1, 128, 40, 40),
           torch.randn(1, 512, 20, 20)]

# Forward pass through your model
out = model(outputs)
for i, o in enumerate(out):
    print(f"i={i + 1} {o.size()}")
