import mxnet as mx

def dilation10(input, num_class):
    ctx_conv1_1 = mx.sym.Convolution(data=input, kernel=(3, 3), pad=(1, 1), num_filter=num_class, name='ctx_conv1_1', workspace=2048)
    ctx_relu1_1 = mx.sym.Activation(data=ctx_conv1_1, act_type='relu', name='ctx_relu1_1')
    ctx_conv1_2 = mx.sym.Convolution(data=ctx_relu1_1, kernel=(3, 3), pad=(1, 1), num_filter=num_class, name='ctx_conv1_2', workspace=2048)
    ctx_relu1_2 = mx.sym.Activation(data=ctx_conv1_2, act_type='relu', name='ctx_relu1_2')

    ctx_conv2_1 = mx.sym.Convolution(data=ctx_relu1_2, kernel=(3, 3), pad=(2, 2), num_filter=num_class, dilate=(2, 2), name='ctx_conv2_1', workspace=2048)
    ctx_relu2_1 = mx.sym.Activation(data=ctx_conv2_1, act_type='relu', name='ctx_relu2_1')

    ctx_conv3_1 = mx.sym.Convolution(data=ctx_relu2_1, kernel=(3, 3), pad=(4, 4), num_filter=num_class, dilate=(4, 4), name='ctx_conv3_1', workspace=2048)
    ctx_relu3_1 = mx.sym.Activation(data=ctx_conv3_1, act_type='relu', name='ctx_relu3_1')

    ctx_conv4_1 = mx.sym.Convolution(data=ctx_relu3_1, kernel=(3, 3), pad=(8, 8), num_filter=num_class, dilate=(8, 8), name='ctx_conv4_1', workspace=2048)
    ctx_relu4_1 = mx.sym.Activation(data=ctx_conv4_1, act_type='relu', name='ctx_relu4_1')

    ctx_conv5_1 = mx.sym.Convolution(data=ctx_relu4_1, kernel=(3, 3), pad=(16, 16), num_filter=num_class, dilate=(16, 16), name='ctx_conv5_1', workspace=2048)
    ctx_relu5_1 = mx.sym.Activation(data=ctx_conv5_1, act_type='relu', name='ctx_relu5_1')

    ctx_conv6_1 = mx.sym.Convolution(data=ctx_relu5_1, kernel=(3, 3), pad=(32, 32), num_filter=num_class, dilate=(32, 32), name='ctx_conv6_1', workspace=2048)
    ctx_relu6_1 = mx.sym.Activation(data=ctx_conv6_1, act_type='relu', name='ctx_relu6_1')

    ctx_conv7_1 = mx.sym.Convolution(data=ctx_relu6_1, kernel=(3, 3), pad=(64, 64), num_filter=num_class, dilate=(64, 64), name='ctx_conv7_1', workspace=2048)
    ctx_relu7_1 = mx.sym.Activation(data=ctx_conv7_1, act_type='relu', name='ctx_relu7_1')

    ctx_fc1 = mx.sym.Convolution(data=ctx_relu7_1, kernel=(3, 3), pad=(1, 1), num_filter=num_class, name='ctx_fc1', workspace=2048)
    ctx_fc1_relu1 = mx.sym.Activation(data=ctx_fc1, act_type='relu', name='ctx_fc1_relu1')

    ctx_final = mx.sym.Convolution(data=ctx_fc1_relu1, kernel=(1, 1), num_filter=num_class, name='ctx_final', workspace=2048)

    return ctx_final


