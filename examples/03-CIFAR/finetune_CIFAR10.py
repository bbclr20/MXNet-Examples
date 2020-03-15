from mxnet import nd, init, cpu, gpu
from mxnet import gluon, autograd
from mxnet.gluon import nn
import mxnet as mx
from mxnet.gluon.data.vision import transforms, datasets 
from gluoncv.data import transforms as gcv_transforms
from gluoncv.model_zoo import get_model
from gluoncv.utils import TrainingHistory
import matplotlib.pyplot as plt
import numpy as np


def test(ctx, val_data):
    metric = mx.metric.Accuracy(name="valid_accuracy")
    for _, batch in enumerate(val_data):
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
        outputs = [net(X) for X in data]
        metric.update(label, outputs)
    return metric.get()

#
# Prepare the dataset
#
cifar_train = datasets.CIFAR10(train=True)
cifar_test = datasets.CIFAR10(train=False)
class_names = [ "airplane", "automobile", "bird", "cat", 
    "deer", "dog", "frog", "horse", "ship", "truck"]

transform_train = transforms.Compose([
    gcv_transforms.RandomCrop(32, pad=4),
    transforms.RandomFlipLeftRight(),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

image, label = cifar_train[0]
plt.figure(class_names[int(label.item())])
plt.imshow(image.asnumpy())
plt.show()

#
# Get GPU num and split the data to each gpu
#
num_gpus = mx.context.num_gpus()
if num_gpus != 0:
    ctx = [mx.gpu(i) for i in range(num_gpus)]
else:
    ctx = cpu()

per_device_batch_size = 128
num_workers = 2
batch_size = per_device_batch_size * num_gpus # laod n batches which will be splited by split_and_load

train_data = gluon.data.DataLoader(
    cifar_train.transform_first(transform_train),
    batch_size=batch_size, shuffle=True, last_batch="rollover", num_workers=num_workers)

val_data = gluon.data.DataLoader(
    cifar_test.transform_first(transform_test),
    batch_size=batch_size, shuffle=False, num_workers=num_workers)

#
# Set the network and training parameters 
#
net = get_model("cifar_resnet20_v1", classes=10) # "classes" is the **kargs of cifar_resnet20_v1
net.initialize(init=init.Xavier(), ctx=ctx)      # use mutiple GPU if len(ctx) != 0
optimizer = 'nag'
optimizer_params = {'learning_rate': 0.1, 'wd': 0.0001, 'momentum': 0.9}
trainer = gluon.Trainer(net.collect_params(), optimizer, optimizer_params)
loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
train_metric = mx.metric.Accuracy(name="train_accuracy")
train_history = TrainingHistory(['training-error', 'validation-error'])

epochs = 15
lr_decay_count = 0

# TODO: use the scheduler 
lr_decay = 0.1
lr_decay_epoch = [80, 160, np.inf]

for epoch in range(epochs):
    train_metric.reset()
    train_loss = 0

    if epoch == lr_decay_epoch[lr_decay_count]:
        trainer.set_learning_rate(trainer.learning_rate*lr_decay)
        lr_decay_count += 1

    for i, batch in enumerate(train_data):
        # Split the data into different GPU
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)

        # Gather the loss from different GPU
        with autograd.record():
            output = [net(X) for X in data]
            loss = [loss_fn(yhat, y) for yhat, y in zip(output, label)]

        for l in loss:
            l.backward()

        trainer.step(batch_size)
        train_loss += sum([l.sum().asscalar() for l in loss])
        train_metric.update(label, output)

    _, train_acc = train_metric.get()
    _, val_acc = test(ctx, val_data)
    train_history.update([1-train_acc, 1-val_acc])
    
    print("Epoch: {:d}, train acc: {:.3f}, test acc: {:.3f}, loss: {:.3f}".format(
        epoch, train_acc, val_acc, train_loss/len(train_data)))

train_history.plot()
net.save_parameters("finetuned_cifar10_resnet20.params")

# next time if you need to use it, just run ??
# net.load_parameters('dive_deep_cifar10_resnet20_v2.params', ctx=ctx)
