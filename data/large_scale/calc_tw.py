import qtensor
from collections import defaultdict
import networkx as nx

p = 4
deg = 3
def bethe_graph(p, degree):
    def add_nodes_to_leafs(graph, deg=3):
        """ Works in-place """
        leaves = [n for n in graph.nodes() if graph.degree(n) <= 1]
        n = graph.number_of_nodes()
        for leaf in leaves:
            next_edges = [(leaf, n+x) for x in range(1, deg)]
            graph.add_edges_from(next_edges)
            n += deg-1
    graph = nx.Graph()
    graph.add_edges_from([(0,1)])
    for i in range(p):
        add_nodes_to_leafs(graph, deg=degree)
    n = graph.number_of_nodes()
    assert n == 2*sum([(deg-1)**y for y in range(p+1)])
    return graph

graph = bethe_graph(p, degree=deg)
print(f'Graph has {graph.number_of_nodes()} nodes')
stats = graph.degree
x = defaultdict(int, {})
for node, deg in stats:
    x[deg] += 1

print(f'Graph degrees stats: {x}')


from qtensor.optimisation.Optimizer import TamakiOptimizer
from qtensor.optimisation import QtreeTensorNet
def optimize_lightcone(self, G, p, edge):
    """
    Builds a circuit for corresponding edge, optimizes it and caches the result
    """
    gamma, beta = [.1]*p, [.2]*p
    circuit = self._edge_energy_circuit(G, gamma, beta, edge=edge)

    tn = QtreeTensorNet.from_qtree_gates(circuit, backend=self.backend)
    print('tensorcnt', len(tn.tensors))
    peo, tn = self.optimizer.optimize(tn)
    width = self.optimizer.treewidth
    # debt: proper opt data needed
    return peo, width

opt = TamakiOptimizer(wait_time=200)
sim = qtensor.QAOAQtreeSimulator(qtensor.DefaultQAOAComposer, optimizer=opt)
peo, width  = optimize_lightcone(sim, graph, p, edge=(0, 1))

print('peo is ', peo)
print('width is', width)
