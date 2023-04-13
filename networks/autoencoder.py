from torch import nn

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, padding=2),
            nn.ReLU(),

            nn.Flatten(1),

            nn.LazyLinear(120),
            nn.ReLU(),
            nn.Linear(120, 80),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(80, 120),
            nn.ReLU(),
            nn.Linear(120, 16 * 160 * 200),
            nn.ReLU(),

            nn.Unflatten(1, (16, 200, 160)),

            nn.ConvTranspose2d(in_channels=16, out_channels=6, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=6, out_channels=6, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=6, out_channels=3, kernel_size=1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


    def encode(self, x):
        x = self.encoder(x)
        return x

    def decode(self, x):
        x = self.decoder(x)
        return x