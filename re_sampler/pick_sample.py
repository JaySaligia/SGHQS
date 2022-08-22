from typing import Tuple, List, Union, Optional
from torch_geometric.typing import PairTensor
from torch import Tensor
import numpy as np
import random
from torch_geometric.utils import degree
from torch_geometric.data import HeteroData
import torch as th


def pick_step(idx_train, y_train, adj_list, size):
    """sample node for training from training graph acoording the probablity
    with respect to degree and label_frequence

    :param idx_train: node in training dataset
    :param y_train: the label according to the training node
    :param adj_list: the adjacency list of training graph
    :param size: the numbers of nodes that can be selected for training
    :return: the ndoes needed to be trained
    """
    # compute the degree of item nodes
    rel_a, rel_b = map(degree, [adj_list[-1][0, :], adj_list[1][1, :]])
    rel_a = th.cat((rel_a, th.Tensor([0])), 0)
    item_degree = rel_a*2 + rel_b*2
    degree_train = item_degree[idx_train]
    label_frequence = (y_train.sum() - len(y_train))*y_train + len(y_train)
    sample_prob = np.array(degree_train) / label_frequence

    return random.choices(idx_train.numpy().tolist(), weights=sample_prob, k=size)


def get_sample_graph(target_nodes, base_graph):
    """Returned the sample graph to train for each batch target nodes

    :param List target_nodes: the nodes to test or valid an induced graph
    :param base_graph: the original graph needed to be subsampled
    :return HeteroData: The induced subgraph for validing or testing
    """
    node_dict_sample = dict.fromkeys(base_graph.node_types)
    if isinstance(target_nodes, list):
        node_dict_sample['item'] = th.tensor(target_nodes)
    else:
        node_dict_sample['item'] = target_nodes.clone().detach()
    for edge_type in base_graph.edge_types:
        src, _, dst = edge_type
        if src == 'item' and dst == 'f':
            continue
        elif dst in ['item', 'f']:
            node_dict_sample[src] = get_one_hop_neighbors(
                                        node_dict_sample[dst],
                                        base_graph[edge_type].edge_index
                                    )

    sample_graph = subgraph_extract(base_graph, node_dict_sample)

    return sample_graph


def get_one_hop_neighbors(target_node, edge_index, flow: str = 'source_to_target'):
    """Returned the one hop neighbors node idx in bipartite subgraph

    :param target_node: the central seed nodes
    :param edge_index: the edge_index of bipartite subgraph
    :param str flow: defaults to 'source_to_target'
    """
    if flow == 'source_to_target':
        num_nodes = edge_index[1].max() + 1
        node_mask = th.zeros(num_nodes, dtype=th.bool)
        node_mask[target_node] = 1
        edge_mask = node_mask[edge_index[1]]
        source_nodes = edge_index[:, edge_mask][0].unique(sorted=False)

    return source_nodes


def bipartite_subgraph(
    subset: Union[PairTensor, Tuple[List[int], List[int]]],
    edge_index: Tensor,
    edge_attr: Optional[Tensor] = None,
    return_edge_mask: bool = False,
) -> Tuple[Tensor, Tensor]:
    """Returned the induced subgraph of the bipartite graph

    :param Union[PairTensor, Tuple[List[int], List[int]]] subset: The nodes to keep
    :param Tensor edge_index: The edge indices
    :param Optional[Tensor] edge_attr: edge features, defaults to None
    :param bool return_edge_mask: defaults to False
    :return Tuple[Tensor, Tensor]: The induced subgraph
    """
    num_nodes = (max(edge_index[0].max().item(), subset[0].max().item()) + 1,
                 max(edge_index[1].max().item(), subset[1].max().item()) + 1)
    node_mask = (th.zeros(num_nodes[0], dtype=th.bool),
                 th.zeros(num_nodes[1], dtype=th.bool))
    node_mask[0][subset[0]] = 1
    try:
        node_mask[1][subset[1]] = 1
    except IndexError:
        print("why")

    edge_mask = node_mask[0][edge_index[0]] & node_mask[1][edge_index[1]]
    edge_index = edge_index[:, edge_mask]
    edge_attr = edge_attr[edge_mask] if edge_attr is not None else None

    # relabel nodes
    node_idx_i = th.zeros(num_nodes[0], dtype=th.long)
    node_idx_j = th.zeros(num_nodes[1], dtype=th.long)
    node_idx_i[node_mask[0]] = th.arange(node_mask[0].sum().item())
    node_idx_j[node_mask[1]] = th.arange(node_mask[1].sum().item())
    edge_index = th.stack([node_idx_i[edge_index[0]],
                           node_idx_j[edge_index[1]]])

    if return_edge_mask:
        return edge_index, edge_attr, edge_mask
    else:
        return edge_index, edge_attr


def subgraph_extract(base_graph, node_dict):
    """Returns the induced subgraph given the node subset

    :param base_graph: the original graph needed to be subsampled
    :param node_dict: a dictionary holding the nodes for each node type
    """
    data = HeteroData()

    for node_type, subset in node_dict.items():
        for key, value in base_graph[node_type].items():
            if key == 'num_nodes':
                data[node_type].num_nodes = subset.size(0)
            elif base_graph[node_type].is_node_attr(key):
                data[node_type][key] = value[subset]
            else:
                data[node_type][key] = value

    for edge_type in base_graph.edge_types:
        src, _, dst = edge_type
        if src not in node_dict or dst not in node_dict:
            continue

        edge_index, _, edge_mask = bipartite_subgraph(
            (node_dict[src], node_dict[dst]),
            base_graph[edge_type].edge_index,
            return_edge_mask=True
        )
        for key, value in base_graph[edge_type].items():
            if key == 'edge_index':
                data[edge_type].edge_index = edge_index
            elif base_graph[edge_type].is_edge_attr(key):
                data[edge_type][key] = value[edge_mask]
            else:
                data[edge_type][key] = value

    return data
