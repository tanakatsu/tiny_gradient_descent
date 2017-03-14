from chainer import Variable, Chain
from chainer import optimizers
from chainer import links as L
import numpy as np

epochs = 10

x = Variable(np.array([5.0], dtype=np.float32).reshape(-1, 1))


class MyChain(Chain):

    def __init__(self):
        super(MyChain, self).__init__(
            b1=L.Bias(1, 1)
        )

    def __call__(self, x):
        y = self.b1(x) * self.b1(x)
        return y


model = MyChain()

optimizer = optimizers.SGD()
optimizer.setup(model)

for i in range(1, epochs + 1):
    x = Variable(np.array([5.0], dtype=np.float32).reshape(-1, 1))

    model.zerograds()
    loss = model(x)
    loss.backward()
    optimizer.update()
    print('epoch=', i, 'loss=', loss.data, 'x=', model.b1(x).data)
