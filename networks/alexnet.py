from torch import nn
import torch


class AlexNet(nn.Module):
    def __init__(self, num_outputs):
        super().__init__()

        self.num_outputs = num_outputs

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Flatten(1),

            nn.LazyLinear(4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_outputs)
        )

    def forward(self, x):
        x = self.layers(x)
        return x

    def predict(self, input):
        output = self.forward(input)

        if self.num_outputs > 1:     # Regression
            pred = torch.argmax(output)
        else:                        # Classification
            pred = output[0][0]

        pred = pred.cpu().detach().numpy()
        return pred
