from chainer import Variable, Chain
from chainer import optimizers
from chainer import links as L
from chainer import functions as F
import numpy as np

epochs = 10

x = Variable(np.array([5.0], dtype=np.float32).reshape(-1, 1))
t = Variable(np.array([0], dtype=np.float32).reshape(-1, 1))


class MyChain(Chain):

    def __init__(self):
        super(MyChain, self).__init__(
            l1=L.Linear(1, 1)
        )

    def __call__(self, x, t):
        y = F.mean_squared_error(self.l1(x), t)
        return y


model = MyChain()

optimizer = optimizers.SGD()
optimizer.setup(model)

for i in range(1, epochs + 1):
    x = Variable(np.array([5.0], dtype=np.float32).reshape(-1, 1))
    t = Variable(np.array([0.0], dtype=np.float32).reshape(-1, 1))

    model.zerograds()
    loss = model(x, t)
    loss.backward()
    optimizer.update()
    print('epoch=', i, 'loss=', loss.data, 'w=', model.l1.W.data, 'b=', model.l1.b.data)
