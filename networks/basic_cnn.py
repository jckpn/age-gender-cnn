from torch import nn
import torch


class BasicCNN(nn.Module):
    def __init__(self, num_outputs):
        super().__init__()

        self.num_outputs = num_outputs

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),

            nn.Flatten(1),

            nn.LazyLinear(num_outputs)
        )
    
    def forward(self, x):
        x = self.layers(x)
        return x
    
    def predict(self, x):
        output = self.forward(x)

        if self.num_outputs > 1:     # Regression
            pred = torch.argmax(output)
        else:                        # Classification
            pred = output[0][0]

        pred = pred.cpu().detach().numpy()
        return pred