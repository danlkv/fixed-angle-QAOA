import numpy as np
import qtensor

def gen_zz_tensor(gamma):
    zz_layers = []
    for g in gamma:
        ZZ = qtensor.OpFactory.ZZ(0, 1, alpha=2*g)
        zz_layers.append(ZZ.gen_tensor())

    for g in reversed(gamma):
        ZZ = qtensor.OpFactory.ZZ(0, 1, alpha=2*g)
        zz_layers.append(ZZ.gen_tensor().conj().T)

    #print('zzl', zz_layers)
    #result = zz_layers.pop()
    #for zz in zz_layers[:]:
    #    result = np.kron(result, zz)

    p = len(gamma)
    ix_per_z = [(i, 2*p+i) for i in range(2*p)]
    result_ixs = list(range(4*p))
    result = np.einsum(*[*sum(zip(zz_layers, ix_per_z), tuple()), result_ixs])
    return result.reshape(2**(2*p), 2**(2*p))

def gen_x_tensor(beta):
    XPhase = qtensor.OpFactory.QtreeFactory.XPhase
    xs = [
        XPhase(0, alpha=2*b_).gen_tensor()
        for b_ in beta
    ]
    for b_ in reversed(beta):
        xd = XPhase(0, alpha=2*b_).gen_tensor().conj().T
        xs.append(xd)

    h = qtensor.OpFactory.QtreeFactory.H(0).gen_tensor()
    h0 = h[:, 0]

    p = len(beta)
    ix_per_x = [(i, i+1) for i in range(2*p)]
    result_ixs = list(range(2*p+1))
    res = np.einsum(*[*sum(zip(xs, ix_per_x), tuple()), h0, [0], h0.conj(), [2*p], result_ixs])
    return res

def gen_x_tensor_traced(beta):
    x = gen_x_tensor(beta)
    p = len(beta)
    return x.sum(p).flatten()

def last_contraction(gamma, beta):
    zz = gen_zz_tensor(gamma)
    x = gen_x_tensor_traced(beta)
    return zz.dot(x)


def recursive_step(p, gamma, beta):
    if p==0:
        return last_contraction(gamma, beta)
    else:
        zz_tensor = gen_zz_tensor(gamma)
        x_tensor = gen_x_tensor_traced(beta)
        t_tensor = recursive_step(p-1, gamma, beta)
        y = t_tensor*x_tensor*t_tensor
        return zz_tensor.dot(y)


Z = np.array([1, -1])
I = np.array([1, 1])

def tree_expectation(p, gamma, beta, observable=Z):
    t_tensor = recursive_step(p-1, gamma, beta)
    zz_tensor = gen_zz_tensor(gamma)
    x_tensor = gen_x_tensor(beta)
    p = len(gamma)
    x_indices = range(2*p+1)
    x_energy = np.einsum(x_tensor, x_indices, observable, [p], range(2*p+1))
    x_tensor = x_energy.sum(p).flatten()

    y = t_tensor*x_tensor*t_tensor

    return y.dot(zz_tensor.dot(y))


def test_tree_expectation():
    p = 3
    gamma, beta = np.random.randn(2, p)
    v = tree_expectation(p, gamma, beta)
    bethe = qtensor.toolbox.bethe_graph(p, 3)
    sim = qtensor.QAOAQtreeSimulator(qtensor.DefaultQAOAComposer)
    v_ref = sim._get_edge_energy(bethe, gamma, beta, (0,1))

if __name__=="__main__":
    test_tree_expectation()

