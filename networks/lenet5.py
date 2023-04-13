from torch import nn
import torch


class LeNet5(nn.Module):
    def __init__(self, num_outputs):
        super().__init__()

        self.num_outputs = num_outputs

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),

            nn.Flatten(1),

            nn.LazyLinear(120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_outputs)
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