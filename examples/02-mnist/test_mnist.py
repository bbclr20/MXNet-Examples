from simplenet import SimpleNet
from mxnet import cpu, gpu, context
from mxnet.gluon.data.vision import transforms
from mxnet import gluon
from mxnet import autograd
from mxnet.metric import Accuracy


transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.13, 0.31),
])


def load_net(param_file="net.params", ctx=cpu(0)):
    net = SimpleNet()
    net.load_parameters(param_file, ctx=ctx)
    return net


def get_val_data(transformer, batch_size=128):
    mnist_valid = gluon.data.vision.FashionMNIST(train=False)
    valid_data = gluon.data.DataLoader(
        mnist_valid.transform_first(transformer),
        batch_size=batch_size, num_workers=4)
    return valid_data


if __name__ == "__main__":
    ctx = gpu(0) if context.num_gpus() else cpu(0)
    net = load_net("net.params", ctx=ctx)
    valid_data = get_val_data(transformer)

    val_acc = Accuracy()
    for data, label in valid_data:
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        with autograd.predict_mode():
            out = net(data)
            val_acc.update(label, out)
    print("Accuray: ", val_acc.get()[1])
