from mxnet import nd
from mxnet import gpu

#
# Basic operations
#
a = nd.array([[1,2,3],[4,5,6]])
print("a: ", a)

b = nd.ones((2,2))
print("b: ", b)

c = nd.full((3,3), 10.0)
print("c: ", c)

d = nd.dot(a.T,a)
print("a.T*a: ", d)

e = d[0:2, 0:2] # slice
print("d[0:2, 0:2]:", d[0:2, 0:2])

#
# Convert to numpy
#
f = e.asnumpy()
print("f: ", f)
print("type(f):", type(f))

#
# Using gpu
#
ctx = gpu(0)
g = e.as_in_context(ctx) # copy the value of e to gpu
print("g: ", g)
