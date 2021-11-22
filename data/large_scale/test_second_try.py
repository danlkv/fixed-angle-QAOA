from second_try import tree_expectation
from pyrofiler import timing
import numpy as np
import qtensor

def test_tree_expectation():
    gamma = [ 1.15963836, -1.62875928,  0.11163931]
    beta =  [-0.84417395, -0.74095452, -0.06684424]
    p = 5
    gamma, beta = np.random.randn(2, p)
    for i in range(3):
        print(f'# try {i}')
        print(gamma, beta)
        with timing('optimum time'):
            v = tree_expectation(p, gamma, beta)
        print('v optimum', v)
        bethe = qtensor.toolbox.bethe_graph(p, 3)
        sim = qtensor.QAOAQtreeSimulator(qtensor.DefaultQAOAComposer)
        with timing('Bucket elimination time'):
            v_ref = sim._get_edge_energy(bethe, gamma, beta, (0,1))
        print('v bucket_elimination', v_ref)
        #assert np.allclose(abs(v), 1)
        assert np.allclose(v, v_ref)

def test_tree_expectation_timings():
    for p in range(2, 13):
        gamma, beta = np.random.randn(2, p)
        with timing(f'Optimum time, p={p}'):
            v = tree_expectation(p, gamma, beta)
