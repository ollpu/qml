import numpy as np
from lib import *

def cost(Y, Yp):
    return np.sum(np.square(Y-Yp))/2

def gradient_step(mp, X, Y, lrate):
    sc = len(X)
    Yp, grad = mp.gradient(X)
    step = np.sum((Yp - Y)*grad, axis=1)
    step /= np.linalg.norm(step)
    step *= -lrate
    nmp = mp.new_with(mp.params + step)
    nmp.cost = cost(Y, Yp)
    return nmp

def learn(mp, X, Y, lrate=0.3, rep=1000):
    X = mp.normalize_input(X)
    Y = np.array(Y, dtype=np.float64)
    best = mp
    for repi in range(rep):
        mp = gradient_step(mp, X, Y, lrate)
        if mp.cost < best.cost: best = mp
    return best
