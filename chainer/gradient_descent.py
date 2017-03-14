import chainer
import numpy as np

epochs = 10
learning_rate = 1e-1
x_val = [5]

for i in range(1, epochs + 1):
    x_data = np.array(x_val, dtype=np.float32)
    x = chainer.Variable(x_data)
    y = x ** 2
    y.backward()
    new_xval = x_val - learning_rate * x.grad
    print('epoch=', i, 'x=', x_val, 'grad=', x.grad, 'new_x=',  new_xval)
    x_val = new_xval
