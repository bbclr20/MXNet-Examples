from mxnet import nd, gpu, autograd
from mxnet.gluon import nn


#
# Compute the differentiation
#
x = nd.array(([1,2,3],[4,5,6]))
x.attach_grad()
print("Before differentiation: ", x.grad)

with autograd.record():
    y = x**2+3
y.backward()
print("After differentiation: ", x.grad)
