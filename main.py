import sys
import json
import matplotlib.pyplot as plt
import numpy as np

from model import Model, ModelParams, UnsetParams
import learner

with open(sys.argv[1]) as jf:
    data = json.load(jf)["TrainingData"]


X = data['Features']
Y = data['Labels']

x, y = zip(*X)

def Xmap(x):
    x = [(x[0]-0.5)*0.8, x[1]-0.2]
    return [x[0], x[1], np.linalg.norm(x)**4, 1]
X = list(map(Xmap, X))

struct = []

rotall = [
    ([], 'X', 0),
    ([], 'Y', 0),
    ([], 'Z', 0),
    ([], 'X', 1),
    ([], 'Y', 1),
    ([], 'Z', 1),
]

struct += rotall
struct.append(([1], 'X', 0))
struct.append(([0], 'Z', 1))
struct.append(([1], 'Y', 0))
struct.append(([0], 'X', 1))
struct.append(([1], 'Z', 0))
struct.append(([0], 'Y', 1))
struct += rotall
struct.append(([0], 'X', 1))
struct.append(([1], 'Z', 0))
struct.append(([0], 'Y', 1))
struct.append(([1], 'Y', 0))
struct.append(([0], 'Z', 1))
struct.append(([1], 'X', 0))
struct += [
    ([], 'X', 0),
    ([], 'Y', 0),
    ([], 'Z', 0),
]

model = Model(2, struct)
# model = Model(1, [
#     ([], 'Y', 0)
# ])


params = UnsetParams()
for repi in range(5):
    tparams = ModelParams(model, 2*np.pi*np.random.rand(len(model.structure)+1))
    tparams.params[-1] = np.random.rand(1)-0.5
    tparams = learner.learn(tparams, X, Y, 0.3, 50)
    if tparams.cost < params.cost: params = tparams
    tparams = learner.learn(tparams, X, Y, 0.1, 100)
    if tparams.cost < params.cost: params = tparams
    tparams = learner.learn(tparams, X, Y, 0.01, 100)
    if tparams.cost < params.cost: params = tparams
    tparams = learner.learn(tparams, X, Y, 0.005, 1000)
    if tparams.cost < params.cost: params = tparams
    print(tparams.cost)
print(params.cost)

Yc = params.classify(X)
print(Yc)

mx, my = np.meshgrid(np.linspace(-1.5, 2.5, 50), np.linspace(-1.5, 1.5, 50))
mf = [Xmap(tx) for tx in zip(mx.flat, my.flat)]
mz = np.reshape(params.predict(mf), mx.shape)

plt.subplot(2, 1, 1)
plt.contourf(mx, my, mz, np.linspace(0, 1, 11), cmap="PiYG")
plt.colorbar()
plt.scatter(x, y, c=Yc)
plt.subplot(2, 1, 2)
plt.scatter(x, y, c=Y)
plt.show()

