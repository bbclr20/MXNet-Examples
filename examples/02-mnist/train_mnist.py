import mxnet as mx
import matplotlib.pyplot as plt
from mxnet import autograd, init, gluon, gpu, cpu, context
from mxnet.gluon.data.vision import datasets, transforms
from simplenet import SimpleNet


text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
               'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']

mnist_train = datasets.FashionMNIST(train=True)

# Check the image and the label
X, y = mnist_train[0]
class_name = text_labels[y]
print("X shape: {}, X dtype:{}".format(X.shape, X.dtype))
print("y: {}({}), y dtype:{}".format(y, class_name, y.dtype))

plt.figure(class_name)
plt.imshow(X.squeeze(axis=2).asnumpy())
plt.show()

# Define the transformation and load the data into batches
transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.13, 0.31),
])

batch_size = 256
train_data = gluon.data.DataLoader(
    mnist_train.transform_first(transformer), batch_size=batch_size, shuffle=True, num_workers=2)

mnist_valid = gluon.data.vision.FashionMNIST(train=False)
valid_data = gluon.data.DataLoader(
    mnist_valid.transform_first(transformer),
    batch_size=batch_size, num_workers=4)

# Set context and start training
ctx = gpu(0) if context.num_gpus() else cpu(0)
print("Using ctx: ", ctx)

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
net = SimpleNet()
net.initialize(init=init.Xavier(), ctx=ctx)
trainer = gluon.Trainer(net.collect_params(), "sgd", {"learning_rate": 0.1})

for epoch in range(10):
    train_loss = 0
    train_acc = mx.metric.Accuracy()
    val_acc = mx.metric.Accuracy()

    for data, label in train_data:
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)

        with autograd.record():
            out = net(data)
            loss = softmax_cross_entropy(out, label)
            train_acc.update(label, out)
        loss.backward()
        trainer.step(batch_size)

        train_loss += loss.mean().asscalar()

    for data, label in valid_data:
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        out = net(data)
        val_acc.update(label, out)

    print("Epoch {:d}: loss {:.3f}, train acc {:.3f}, valid acc: {:.3f}".format(
            epoch, train_loss/len(train_data), train_acc.get()[1], val_acc.get()[1]))

# save the model
file_name = "net.params"
net.save_parameters(file_name)
