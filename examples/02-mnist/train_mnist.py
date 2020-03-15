from mxnet import nd, autograd, init, gluon, gpu, cpu, context
from mxnet.gluon import nn
from mxnet.gluon.data.vision import datasets, transforms
import matplotlib.pyplot  as plt


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
            nn.Dense(10, activation="relu"),
        )

    def forward(self, x):
        return self.net(x)

def acc(output, label):
    return (output.argmax(axis=1) == label.astype('float32')).mean().asscalar()

text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
               'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']

mnist_train = datasets.FashionMNIST(train=True)

#
# Check the image and the label
#
X, y = mnist_train[0]
class_name = text_labels[y]
print("X shape: {}, X dtype:{}".format(X.shape, X.dtype))
print("y: {}({}), y dtype:{}".format(y, class_name, y.dtype))

plt.figure(class_name)
plt.imshow(X.squeeze(axis=2).asnumpy())
plt.show()

#
# Define the transformation and load the data into batches
#
transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.13,0.31),
])

batch_size = 256
train_data = gluon.data.DataLoader(
    mnist_train.transform_first(transformer), batch_size=batch_size, shuffle=True, num_workers=2)

mnist_valid = gluon.data.vision.FashionMNIST(train=False)
valid_data = gluon.data.DataLoader(
    mnist_valid.transform_first(transformer),
    batch_size=batch_size, num_workers=4)

#
# Set context and start training
#
if context.num_gpus()!= 0:
    ctx = gpu(0)
else:
    ctx = cpu(0)
print("Using ctx: ", ctx)

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
net = SimpleNet()
net.initialize(init=init.Xavier(), ctx=ctx)
trainer = gluon.Trainer(net.collect_params(), "sgd", {"learning_rate":0.1})

for epoch in range(10):
    train_loss, train_acc, valid_acc = 0.0, 0.0, 0.0
    for data, label in train_data:
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)

        with autograd.record():
            out = net(data)
            loss = softmax_cross_entropy(out, label)
        loss.backward()
        trainer.step(batch_size)

        train_loss += loss.mean().asscalar()
        train_acc += acc(out, label)
    
    for data, label in valid_data:
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        out = net(data)
        valid_acc += acc(out, label)

    print("Epoch {:d}: loss {:.3f}, train acc {:.3f}, valid acc: {:.3f}".format(
            epoch, train_loss/len(train_data), train_acc/len(train_data), valid_acc/len(valid_data)))
