from mxnet.gluon import nn


class SimpleNet(nn.Block):
    def __init__(self, **kargs):
        super(SimpleNet, self).__init__()
        self.net = nn.Sequential()
        self.net.add(
            nn.Conv2D(16, 5, activation="relu"),
            nn.MaxPool2D(2, 2),
            nn.Conv2D(23, 3, activation="relu"),
            nn.MaxPool2D(2, 2),
            nn.Flatten(),
            nn.Dense(120, activation="relu"),
            nn.Dense(84, activation="relu"),
            nn.Dense(10),
        )

    def forward(self, x):
        return self.net(x)