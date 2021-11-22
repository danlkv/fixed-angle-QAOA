import numpy as np
import qtensor

def gen_zz_tensor(gamma):
    zz_layers = []
    p = len(gamma)
    for g in gamma:
        ZZ = qtensor.OpFactory.ZZ(0, 1, alpha=2*g)
        zz_layers.append(ZZ.gen_tensor())

    ix_per_z = [(i, p+i) for i in range(p)]
    result_ixs = list(range(2*p))
    result = np.einsum(*[*sum(zip(zz_layers, ix_per_z), tuple()), result_ixs])
    return result.reshape(2**(p), 2**(p))

def gen_zz_unit(gamma):
    a = gen_zz_tensor(gamma)
    b = gen_zz_tensor(-np.flip(gamma))
    return a, b

def gen_x_tensor(beta):
    XPhase = qtensor.OpFactory.QtreeFactory.XPhase
    xs = [
        XPhase(0, alpha=2*b_).gen_tensor()
        for b_ in beta
    ]

    h = qtensor.OpFactory.QtreeFactory.H(0).gen_tensor()
    h0 = h[:, 0]

    p = len(beta)
    ix_per_x = [(i, i+1) for i in range(p)]
    result_ixs = list(range(p+1))
    res = np.einsum(*[*sum(zip(xs, ix_per_x), tuple()), h0, [0], result_ixs])
    return res

def gen_x_unit(beta):
    a = gen_x_tensor(beta)
    # careful: works only because Hdag = H
    p = len(beta)
    b = gen_x_tensor(-np.array(beta))
    ixa = list(range(p+1))
    ixb = list(range(2*p, p-1, -1))
    result_ixs = list(range(2*p+1))
    result_ixs.remove(p)
    r = np.einsum(a, ixa, b, ixb, result_ixs)
    rs = r.reshape(2**(p), 2**(p))
    return rs

def gen_x_energy(beta, observable):
    a = gen_x_tensor(beta)
    # careful: works only because Hdag = H
    p = len(beta)
    b = gen_x_tensor(-np.array(beta))
    ixa = list(range(p+1))
    ixb = list(range(2*p, p-1, -1))
    result_ixs = list(range(2*p+1))
    result_ixs.remove(p)
    r = np.einsum(a, ixa, b, ixb, observable, [p], result_ixs)
    rs = r.reshape(2**(p), 2**(p))
    return rs

def last_contraction(gamma, beta):
    za, zb = gen_zz_unit(gamma)
    x = gen_x_unit(beta)
    return za.dot(x.dot(zb.T))


def recursive_step(p, gamma, beta):
    if p==0:
        return last_contraction(gamma, beta)
    else:
        za, zb = gen_zz_unit(gamma)
        x_tensor = gen_x_unit(beta)
        t_tensor = recursive_step(p-1, gamma, beta)
        y = t_tensor*x_tensor*t_tensor
        return za.dot(y.dot(zb.T))


Z = np.array([1, -1])
I = np.array([1, 1])

def tree_expectation(p, gamma, beta, observable=Z):
    t_tensor = recursive_step(p-1, gamma, beta)
    za, zb = gen_zz_unit(gamma)
    p = len(gamma)
    x_tensor = gen_x_energy(beta, observable)

    y = t_tensor*x_tensor*t_tensor

    m = za.dot(y.dot(zb.T))
    print('m.size', m.size, 'm.memory', m.size*128/8/1e9, 'GB')
    return np.sum(m*y)

def test_tree_expectation():
    p = 3
    gamma, beta = np.random.randn(2, p)
    v = tree_expectation(p, gamma, beta)
    bethe = qtensor.toolbox.bethe_graph(p, 3)
    sim = qtensor.QAOAQtreeSimulator(qtensor.DefaultQAOAComposer)
    v_ref = sim._get_edge_energy(bethe, gamma, beta, (0,1))

if __name__=="__main__":
    test_tree_expectation()

