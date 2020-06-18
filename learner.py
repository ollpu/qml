import numpy as np
from lib import *

cost_power = 10

def cost(Y, Yp):
    return np.sum(np.power(Y-Yp, cost_power))/cost_power

def cost_gradient(Y, Yp):
    return np.power(Yp - Y, cost_power-1)

def gradient(mp, X, Y):
    sc = len(X)
    Yp, grad = mp.gradient(X)
    step = np.sum(cost_gradient(Y, Yp)*grad, axis=1)
    step /= np.linalg.norm(step)
    return step, Yp

def learn(mp, X, Y, lrate=0.3, rep=1000, beta1=0.9, beta2=0.999, eps=1e-4):
    X = mp.normalize_input(X)
    Y = np.array(Y, dtype=np.float64)
    best = mp
    m = np.zeros_like(mp.params)
    v = np.zeros_like(mp.params)
    for repi in range(rep):
        step, Yp = gradient(mp, X, Y)
        
        # Adam optimizer
        m = beta1 * m + (1 - beta1) * step
        v = beta2 * v + (1 - beta2) * np.power(step, 2)
        m_hat = m / (1 - np.power(beta1, repi+1))
        v_hat = v / (1 - np.power(beta2, repi+1))
        step = m_hat / (np.sqrt(v_hat) + eps)
        
        mp = mp.new_with(mp.params - lrate * step)
        mp.cost = cost(Y, Yp)
        if mp.cost < best.cost: best = mp
    return best
