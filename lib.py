import numpy as np
import functools

p_dtype = np.complex128

def kp(e):
    return functools.reduce(np.kron, e)

I = np.array([[1, 0], [0, 1]], dtype=p_dtype)
Z = np.array([[1, 0], [0, 0]], dtype=p_dtype)
O = np.array([[0, 0], [0, 1]], dtype=p_dtype)

def IU(n):
    return np.identity(2**n, dtype=p_dtype)

def rot_gate(pauli, theta, diff=False):
    if diff:
        return rot_gate(pauli, theta+np.pi)/2
    else:
        sint = np.sin(theta/2)
        cost = np.cos(theta/2)
        if pauli == 'X':
            mat = [[cost, -1j*sint], [-1j*sint, cost]]
        elif pauli == 'Y':
            mat = [[cost, -sint], [sint, cost]]
        elif pauli == 'Z':
            mat = [[np.exp(-0.5j*theta), 0], [0, np.exp(0.5j*theta)]]
        return np.array(mat, p_dtype)
