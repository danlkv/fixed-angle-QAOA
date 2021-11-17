from first_try import gen_x_tensor, gen_zz_tensor, tree_expectation, gen_x_tensor_traced
import numpy as np
import qtensor

def test_zz_gen():
    gamma = [0, 0.123]
    zz = gen_zz_tensor(gamma)
    p = len(gamma)
    assert zz.size == 16**p
    assert np.allclose(zz.T, zz)
    print('zz gen shape', zz.shape)
    assert np.allclose(zz[0,0]+zz[-1,-1], 2)
    ones = np.zeros(4**p)*0j
    h = qtensor.OpFactory.QtreeFactory.H(0).gen_tensor()
    h0 = h[:, 0]
    ones[0] = h0[0]*h0.conj()[0]
    ones[-1] =h0[1]*h0.conj()[1]
    f = ones.dot(zz.dot(ones))
    print('f', f)
    assert np.allclose(f, 1)

def test_gen_x_tensor():
    # asserts flat x tensor
    beta = [0.05, 0.]
    p = len(beta)
    x = gen_x_tensor(beta)
    assert len(x.shape) == 2*p+1
    assert x.shape[0] == 2
    print('Super-x tensor', x.round(2))
    #assert np.allclose(x.flatten(), [0, 1, 0,0, 0,0, 1,0])
    assert np.allclose(x.sum(), 1)
    h = qtensor.OpFactory.QtreeFactory.H(0).gen_tensor()
    h0 = h[:, 0]
    hh0 = np.outer(h0, h0)
    xs = x.sum(axis=(1, 2, 3))
    print('xs', xs)
    print('hh0', hh0)
    print('hh0e', hh0*np.eye(2))
    assert np.allclose(xs, np.eye(2)*hh0)

    # 
    x = gen_x_tensor_traced([0, 0])
    ones = np.zeros(4**p)*0j
    h = qtensor.OpFactory.QtreeFactory.H(0).gen_tensor()
    h0 = h[:, 0]
    ones[0] = h0[0]*h0.conj()[0]
    ones[-1] =h0[1]*h0.conj()[1]
    assert np.allclose(x, ones)

def test_tree_expectation():
    gamma = [ 1.15963836, -1.62875928,  0.11163931]
    beta =  [-0.84417395, -0.74095452, -0.06684424]
    p = 5
    gamma, beta = np.random.randn(2, p)
    print(gamma, beta)
    v = tree_expectation(p, gamma, beta)
    print('v', v)
    bethe = qtensor.toolbox.bethe_graph(p, 3)
    sim = qtensor.QAOAQtreeSimulator(qtensor.DefaultQAOAComposer)
    v_ref = sim._get_edge_energy(bethe, gamma, beta, (0,1))
    print('v_ref', v_ref)
    #assert np.allclose(abs(v), 1)
    assert np.allclose(v, v_ref)
