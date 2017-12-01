import mxnet as mx
import numpy as np
from collections import namedtuple

x = mx.nd.ones((100,100))
y = mx.nd.ones((100,100))

data = mx.sym.var('data')
fc1 = mx.symbol.FullyConnected(data, name='fc1', num_hidden=128)
act1 = mx.symbol.Activation(fc1, name='relu1', act_type='relu')
fc2 = mx.symbol.FullyConnected(act1, name='fc2', num_hidden=128)
act2 = mx.symbol.Activation(fc2, name='relu2', act_type='relu')
fc3 = mx.symbol.FullyConnected(act2, name='fc3', num_hidden=100) # TODO: output shape
sym = fc3

mod = mx.mod.Module(symbol=fc3, label_names=None)
mod.bind(data_shapes=[('data', x[0].shape)]) # inputs_need_grad=True # TODO: data shape
initializer = mx.init.Xavier(rnd_type='gaussian', factor_type='in', magnitude=2)
lr = 0.5
mom = 0.5
wd = 0.0
optimizer_params = {
    'learning_rate'     : lr,
    'momentum'          : mom,
    'wd'                : wd,
    }
mod.init_params(initializer)
mod.init_optimizer(optimizer_params=optimizer_params, force_init=True)

pre_err = []
while True:

    err = 0
    for i in range(x.shape[0]):

        Batch = namedtuple('Batch', ['data'])

        mod.forward(Batch(mx.nd.array(x[i])))
        output = mod.get_outputs()[0]

        grad = 2 * (output - y[i])

        mod.backward([grad])
        mod.update()

        err += mx.nd.norm(output - y[i]).asscalar()

    if len(pre_err) != 0:
        if err > pre_err[-1]:
            print 'changing learning rate'
            lr /= 2.0
            mom /= 2.0
            optimizer_params = {
                'learning_rate'     : lr,
                'momentum'          : mom,
                'wd'                : wd
                }
            mod.init_optimizer(optimizer_params=optimizer_params, force_init=True)
        if len(pre_err) == 10:
            if all(x == err for x in pre_err[-10:]):
                break
            pre_err = pre_err[1:]
    pre_err.append(err)

    print err
