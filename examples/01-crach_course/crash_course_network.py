from mxnet import nd, gpu
from mxnet.gluon import nn


#
# Create a Dense layer
#
x = nd.ones((3, 2))
net = nn.Dense(2) # 2 is the # of output channels
net.initialize()
y = net(x)
print("{} {} {}".format("="*20, "Dense layer", "="*20))
print("y: ", net(x))
print("net weight: ", net.weight.data())
print("net bias: ", net.bias.data())

#
# Create a sequential layers
#
net = nn.Sequential()
net.add(
    nn.Dense(4),
    nn.Dense(6),
)
net.initialize()
y = net(x)
print("{} {} {}".format("="*20, "Sequential layers", "="*20))
print("y: {}, y.shape: {}".format(net(x), y.shape))
print("net[0].weight.data(): {}, net[0].weight.shape: {}".format(net[0].weight.data(), net[0].weight.shape))
print("net[1].weight.data(): {}, net[1].weight.shape: {}".format(net[1].weight.data(), net[1].weight.shape))


#
# Using a class
#
class SimpleNet(nn.Block):

    def __init__(self, **kwargs):
        super(SimpleNet, self).__init__(**kwargs)
        self.body = nn.Sequential()
        self.body.add(
            nn.Dense(4),
            nn.Dense(6),
        )
        self.dense = nn.Dense(3)

    def forward(self, x):
        x = nd.relu(self.body(x))
        return self.dense(x)

net = SimpleNet()
net.initialize()
print("{} {} {}".format("="*20, "class", "="*20))
print("net(x).shape: ", net(x).shape)
print("net.body[0].weight.data(): ", net.body[0].weight.data())

#
# Using gpu
#
ctx = gpu(0)
x = x.as_in_context(ctx)
gpu_net = SimpleNet()
gpu_net.initialize(ctx=ctx)
y = gpu_net(x)
print("{} {} {}".format("="*20, "Using GPU", "="*20))
print(y)