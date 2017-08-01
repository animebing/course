from __future__ import print_function
import mxnet as mx
import numpy as np

data = mx.sym.Variable('data')
f = 2  # f is the upsample factor
upsample = mx.sym.Deconvolution(data=data, kernel=(2*f, 2*f), stride=(f, f), pad=(f//2, f//2), num_filter=3, num_group=3, name="upsample")  # here the shape of data is (1, 3, 2, 2)
print("arguments list: ", upsample.list_arguments()) # there is 'data', 'upsample_weight', no 'upsample_bias'

exe = upsample.simple_bind(ctx=mx.gpu(0), data=(1, 3, 2, 2))
exe.arg_dict['data'][:] = mx.nd.array(np.random.randn(1, 3, 2, 2))
print("data: ", exe.arg_dict['data'].asnumpy())
init = mx.init.Initializer()
init._init_bilinear('upsample_weight', exe.arg_dict['upsample_weight'])
exe.forward()
exe.outputs[0].wait_to_read()
print("upsample data: ", exe.outputs[0].asnumpy()) # the shape is (1, 2, 6, 6)

