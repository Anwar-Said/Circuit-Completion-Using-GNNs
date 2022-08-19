import numpy as np
import networkx as nx
from PySpice.Spice.Parser import SpiceParser
from PySpice.Spice import BasicElement
from PySpice.Spice.Netlist import Node
import os
import torch
import deepsnap.graph

component_types = [
    'unknown',
    BasicElement.Resistor,
    BasicElement.BehavioralCapacitor,
    BasicElement.VoltageSource,
    BasicElement.Mosfet,
    BasicElement.SubCircuitElement,
    Node,
    BasicElement.Diode,
    BasicElement.BehavioralInductor,
    BasicElement.CurrentSource,
    BasicElement.VoltageControlledCurrentSource,
    BasicElement.VoltageControlledVoltageSource,
    BasicElement.Capacitor,
    BasicElement.CoupledInductor,
    BasicElement.JunctionFieldEffectTransistor,
    BasicElement.BipolarJunctionTransistor,
    BasicElement.XSpiceElement,
    BasicElement.BehavioralSource,
    BasicElement.SemiconductorResistor,
    BasicElement.Mesfet,
    BasicElement.Inductor,
    BasicElement.CurrentControlledVoltageSource,
    BasicElement.UniformDistributedRCLine,
    BasicElement.CoupledMulticonductorLine,
    BasicElement.SingleLossyTransmissionLine,
    BasicElement.BehavioralResistor,
    BasicElement.LossyTransmission,
    BasicElement.CurrentControlledCurrentSource,
]

subcircuit_types = {}
script_dir = os.path.dirname(os.path.realpath(__file__))
# FIXME: uncomment below to re-enable labels for subcircuits
# with open(os.path.join(script_dir, 'subcircuit-types.txt'), 'r') as f:
    # subcircuit_label_pairs = (line.split(' ') for line in f if len(line.split(' ')) == 2)
    # for (subcircuit, label) in subcircuit_label_pairs:
        # label = label.strip()
        # subcircuit_types[subcircuit] = label
        # if label not in component_types:
            # component_types.append(label)

def get_component_type_index(element):
    element_type = type(element)
    if element_type is BasicElement.SubCircuitElement:
        element_type = subcircuit_types.get(element.subcircuit_name, element_type)

    return component_types.index(element_type)


def components(circuit):
    component_list = []
    for element in circuit.elements:
        if element not in component_list:
            component_list.append(element)

        nodes = [ pin.node for pin in element.pins ]
        for node in nodes:
            if node not in component_list:
                component_list.append(node)
    return component_list


def netlist_as_graph(textfile):
    parser = SpiceParser(source=textfile)
    circuit = parser.build_circuit()
    component_list = components(circuit)
    adj = {}

    for element in circuit.elements:
        element_id = component_list.index(element)

        if element_id not in adj:
            adj[element_id] = []

        nodes = [ pin.node for pin in element.pins ]
        node_ids = [component_list.index(node) for node in nodes]
        adj[element_id].extend(node_ids)

        for node_id in node_ids:
            if node_id not in adj:
                adj[node_id] = []
            adj[node_id].append(element_id)
    
    g = nx.Graph(nx.from_dict_of_lists(adj))
    return component_list, g

def get_nodes_edges(circuit):
    component_list = components(circuit)
    edges = []

    for element in circuit.elements:
        element_id = component_list.index(element)

        nodes = [ pin.node for pin in element.pins ]
        node_ids = [component_list.index(node) for node in nodes]
        edges.extend([ (element_id, node_id, {'pin': i}) for (i, node_id) in enumerate(node_ids) ])

    nodes = [ (i, {'component': component}) for (i, component) in enumerate(component_list) ]
    return nodes, edges

def valid_netlist_sources(files):
    netlists = ( open(f, 'rb').read().decode('utf-8', 'ignore') for f in files )
    return ( text for text in netlists if is_valid_netlist(text) )

def is_valid_netlist(textfile, name=None):
    try:
        parser = SpiceParser(source=textfile)
        circuit = parser.build_circuit()
        return len(circuit.elements) > 0
    except:
        if name:
            print(f'invalid spice file: {name}', file=sys.stderr)
        return False

def component_index_name(idx):
    component = component_types[idx]
    if type(component) is not str:
        return component.__name__
    return component

def ensure_no_nan(tensor):
    nan_idx = torch.isnan(tensor).nonzero(as_tuple=True)
    nan_count = nan_idx[0].shape[0]
    assert nan_count == 0, 'nodes contain nans'

def to_networkx(dataset):
    graphs = []
    for sgraph in dataset:
        node_count = sgraph.x.shape[0]
        nodes = ( (i, {'node_feature': torch.tensor(sgraph.x[i])}) for i in range(node_count) )
        graph = nx.Graph()
        graph.add_nodes_from(nodes)

        row_idx, col_idx = sgraph.a.nonzero()
        edges = list(zip(row_idx, col_idx))
        graph.add_edges_from(edges)
        edge_count = len(graph.edges)
        if 2 * edge_count != len(edges):
            print('edges', edges)
            print('(sorted) edges:')
            print(sorted([ sorted(edge) for edge in edges], key=lambda p: p[0] + p[1]/100))
            print('graph.edges:')
            print(sorted([ sorted(edge) for edge in graph.edges], key=lambda p: p[0] + p[1]/100))
            #print(graph.edges)

        print('expected count:', len(edges), f'({2 * edge_count})')
        assert 2 * edge_count == len(edges), f'Expected {len(edges)} edges. Found {edge_count}'
        graphs.append(graph)

    return graphs

def to_deepsnap(dataset):
    if 'to_deepsnap' in dir(dataset):
        return dataset.to_deepsnap()

    graphs = []
    nxgraphs = to_networkx(dataset)
    src_graphs = zip((sgraph for sgraph in dataset), nxgraphs)

    for (sgraph, nxgraph) in src_graphs:
        node_features = torch.tensor(sgraph.x)
        ensure_no_nan(node_features)

        if sgraph.y is not None:
            label = torch.tensor([sgraph.y.argmax()])
            deepsnap.graph.Graph.add_graph_attr(nxgraph, 'graph_label', label)

        graphs.append(deepsnap.graph.Graph(nxgraph))

    return graphs
