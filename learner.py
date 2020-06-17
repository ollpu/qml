import numpy as np
from lib import *
from model import UnsetParams

def cost(Y, Yp):
    return np.sum(np.square(Y-Yp))/2

def gradient_step(mp, X, Y, lrate):
    sc = len(X)
    Yp, grad = mp.gradient(X)
    step = np.sum((Yp - Y)*grad, axis=1)
    step /= np.linalg.norm(step)
    step *= -lrate
    return mp.new_with(mp.params + step, cost(Y, Yp))

def learn(mp, X, Y, lrate=0.3, rep=1000):
    X = mp.normalize_input(X)
    best = UnsetParams()
    for repi in range(rep):
        mp = gradient_step(mp, X, Y, lrate)
        if mp.cost < best.cost: best = mp
    return best
