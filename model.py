import numpy as np
from lib import *

# Port:
# ([ctrls], 'X/Y/Z', target)

class Model:
    def __init__(self, n, structure):
        # Number of qubits
        self.n = n
        # Array of ports
        self.structure = structure


class ModelParams:
    def __init__(self, model, params=None, cost=np.inf):
        self.model = model
        self.n = model.n
        self.dc = 2**model.n
        # bias @ [-1]
        self.count = len(model.structure)+1
        self.params = params if params is not None else np.zeros(self.count)
        self.cost = cost

    def new_with(self, params, cost=None):
        return ModelParams(self.model, params, cost)

    def copy(self):
        return ModelParams(self.model, self.params.copy(), self.cost)

    def normalize_input(self, st):
        st = np.array(st, np.complex128)
        dc_in = st.shape[-1]
        if dc_in < self.dc:
            if len(st.shape) == 1:
                st = np.pad(st, (0, self.dc-dc_in), mode='constant')
            else:
                st = np.pad(st, ((0, 0), (0, self.dc-dc_in)), mode='constant')
        if len(st.shape) == 1:
            st /= np.linalg.norm(st)
        else:
            st /= np.linalg.norm(st, axis=-1)[:,None]
        return st

    def predict(self, X):
        st = self.normalize_input(X)
        for port, param in zip(self.model.structure, self.params):
            gate = rot_gate(port[1], param)
            op = [I]*self.n
            for i in port[0]:
                op[i] = O
            op[port[2]] = gate - I
            mat = IU(self.n) + kp(op)
            st = (mat @ st.T).T
        p1 = np.square(np.linalg.norm(st[..., self.dc//2:], axis=-1))
        return p1 + self.params[-1]

    def classify(self, X):
        return (self.predict(X) > 0.5).astype(int)

    def gradient(self, X):
        st = self.normalize_input(X)
        unitaries = []
        unitaries_diff = []
        for port, param in zip(self.model.structure, self.params):
            gate = rot_gate(port[1], param)
            gate_diff = rot_gate(port[1], param, True)
            op = [I]*self.n
            for i in port[0]:
                op[i] = O
            op[port[2]] = gate_diff
            mat = kp(op)
            unitaries_diff.append((mat @ st.T).T)

            op[port[2]] = gate - I
            mat = IU(self.n) + kp(op)
            st = (mat @ st.T).T
            unitaries.append(mat)
        # Gradient of last layer (d res / d state_i).
        # The compontents of the complex numbers are separate partial derivatives
        # of their effect on the result.
        gr_l = 2*st
        gr_l[..., :self.dc//2] = 0
        result = []
        for u, ud in zip(reversed(unitaries), reversed(unitaries_diff)):
            result.append(np.real(np.sum(ud*np.conj(gr_l), axis=-1)))
            gr_l = np.conj(np.conj(gr_l) @ u)
        result.reverse()
        result.append(np.full(st.shape[-2:-1], 1.))
        result = np.array(result)
        p1 = np.square(np.linalg.norm(st[..., self.dc//2:], axis=-1))
        return (p1 + self.params[-1], result)

class UnsetParams:
    def __init__(self):
        self.cost = np.inf

