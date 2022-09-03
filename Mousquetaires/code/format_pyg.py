import argparse
from pydoc import describe
from tkinter import W
from torch_geometric.data import HeteroData
import torch
import numpy as np
from tqdm import tqdm
import os.path as osp
import pickle as pkl

edge_size = 0
node_size = 0


def read_node_atts(node_file, pyg_file, label_file=None):
    node_maps = {}
    node_embeds = {}
    count = 0
    lack_num = {}
    node_counts = node_size
    if osp.exists(pyg_file + ".nodes.pyg") == False:
        print("Start loading node information")
        process = tqdm(total=node_counts)
        with open(node_file, 'r') as rf:
            while True:
                line = rf.readline()  # node_id int, node_type string, node_atts string (notice that node_atts are 256-dimensional feature vector strings with delimiter ":")
                if line is None or len(line) == 0:
                    break
                info = line.strip().split(",")

                node_id = int(info[0])
                node_type = info[1].strip()

                node_maps.setdefault(node_type, {})  # 没有这个type则新建对应的value为{}
                node_id_v2 = len(node_maps[node_type])  # 在改type下的新版node_id
                node_maps[node_type][node_id] = node_id_v2

                node_embeds.setdefault(node_type, {})
                lack_num.setdefault(node_type, 0)
                if node_type == 'item':  # 商品
                    if len(info[2]) < 50:
                        node_embeds[node_type][node_id_v2] = np.zeros(256, dtype=np.float32)
                        lack_num[node_type] += 1
                    else:
                        node_embeds[node_type][node_id_v2] = np.array([x for x in info[2].split(":")], dtype=np.float32)
                else:
                    if len(info[2]) < 50:
                        node_embeds[node_type][node_id_v2] = np.zeros(256, dtype=np.float32)
                        lack_num[node_type] += 1  # 缺少特征embedding的节点数量
                    else:
                        node_embeds[node_type][node_id_v2] = np.array([x for x in info[2].split(":")], dtype=np.float32)

                count += 1
                if count % 100000 == 0:
                    process.update(100000)

        process.update(node_size % 100000)
        process.close()
        print("Complete loading node information\n")

        print("Num of total nodes:", count)
        print('Node_types:', list(node_maps.keys()))
        print('Node_type Num Num_lack_feature:')
        for node_type in node_maps:
            print(node_type, len(node_maps[node_type]), lack_num[node_type])

        labels = []
        if label_file is not None:
            labels_info = [x.strip().split(",") for x in open(label_file).readlines()]
            for i in range(len(labels_info)):
                x = labels_info[i]
                item_id = node_maps['item'][int(x[0])]  # node_id_v2
                label = int(x[1])
                labels.append([item_id, label])

        nodes_dict = {'maps': node_maps, 'embeds': node_embeds}
        nodes_dict['labels'] = {}
        nodes_dict['labels']['item'] = labels
        print('\n')
        print('Start saving pkl-style node information')
        pkl.dump(nodes_dict, open(pyg_file + ".nodes.pyg", 'wb'), pkl.HIGHEST_PROTOCOL)
        print('Complete saving pkl-style node information\n')

    else:
        nodes = pkl.load(open(pyg_file + ".nodes.pyg", 'rb'))
        node_embeds = nodes['embeds']
        node_maps = nodes['maps']
        labels = nodes['labels']['item']

    graph = HeteroData()

    print("Start converting into pyg data")  # 构建Pyg的Heterogeneous Graph
    for node_type in tqdm(node_embeds, desc="Node features, numbers and mapping"):
        graph[node_type].x = torch.empty(len(node_maps[node_type]), 256)
        for nid, embedding in tqdm(node_embeds[node_type].items()):
            graph[node_type].x[nid] = torch.from_numpy(embedding)
        graph[node_type].num_nodes = len(node_maps[node_type])
        graph[node_type].maps = node_maps[node_type]

    if label_file is not None:
        graph['item'].y = torch.zeros(len(node_maps['item']), dtype=torch.long) - 1
        for index, label in tqdm(labels, desc="Node labels"):
            graph['item'].y[index] = label  # 为每个item分配一个label

        indices = (graph['item'].y != -1).nonzero().squeeze()
        print("Num of true labeled nodes:{}".format(indices.shape[0]))
        train_val_random = torch.randperm(indices.shape[0])
        train_idx = indices[train_val_random][:int(indices.shape[0] * 0.8)]
        val_idx = indices[train_val_random][int(indices.shape[0] * 0.8):]
        graph['item'].train_idx = train_idx
        graph['item'].val_idx = val_idx

    for ntype in graph.node_types:
        graph[ntype].n_id = torch.arange(graph[ntype].num_nodes)
    print("Complete converting into pyg data\n")

#    print("Start saving into pyg data")
#    torch.save(graph, pyg_file + ".pt")
#    print("Complete saving into pyg data\n")
    return graph


def format_pyg_graph(edge_file, node_file, pyg_file, label_file=None):
    if osp.exists(pyg_file + ".pt") and args.reload == False:
#        graph = torch.load(pyg_file + ".pt")
        print("PyG graph of " + ("session2" if "session2" in pyg_file else "session1") + " has generated")
        return 0
    else:
        print("##########################################")
        print("### Start generating PyG graph of " + ("session2" if "session2" in args.storefile else "session1"))
        print("##########################################\n")
        graph = read_node_atts(node_file, pyg_file, label_file)

    print("Start loading edge information")
    process = tqdm(total=edge_size)
    edges = {}
    count = 0
    with open(edge_file, 'r') as rf:
        while True:
            line = rf.readline()
            if line is None or len(line) == 0:
                break
            line_info = line.strip().split(",")
            source_id, dest_id, source_type, dest_type, edge_type = line_info
            source_id = graph[source_type].maps[int(source_id)]  # source_type下的新的source_id
            dest_id = graph[dest_type].maps[int(dest_id)]
            edges.setdefault(edge_type, {})
            edges[edge_type].setdefault('source', []).append(int(source_id))
            edges[edge_type].setdefault('dest', []).append(int(dest_id))
            edges[edge_type].setdefault('source_type', source_type)
            edges[edge_type].setdefault('dest_type', dest_type)
            count += 1
            if count % 100000 == 0:
                process.update(100000)
    process.update(edge_size % 100000)
    process.close()
    print('Complete loading edge information\n')

    print('Start converting edge information')
    for edge_type in edges:
        source_type = edges[edge_type]['source_type']
        dest_type = edges[edge_type]['dest_type']
        source = torch.tensor(edges[edge_type]['source'], dtype=torch.long)
        dest = torch.tensor(edges[edge_type]['dest'], dtype=torch.long)
        graph[(source_type, edge_type, dest_type)].edge_index = torch.vstack([source, dest])

    for edge_type in [('b', 'A_1', 'item'),
                      ('f', 'B', 'item'),
                      ('a', 'G_1', 'f'),
                      ('f', 'G', 'a'),
                      ('a', 'H_1', 'e'),
                      ('f', 'C', 'd'),
                      ('f', 'D', 'c'),
                      ('c', 'D_1', 'f'),
                      ('f', 'F', 'e'),
                      ('item', 'B_1', 'f'),
                      ('item', 'A', 'b'),
                      ('e', 'F_1', 'f'),
                      ('e', 'H', 'a'),
                      ('d', 'C_1', 'f')]:
        temp = graph[edge_type].edge_index
        del graph[edge_type]
        graph[edge_type].edge_index = temp

    print('Complete converting edge information\n')
    print('Start saving into pyg data')
    torch.save(graph, pyg_file + ".pt")
    print('Complete saving into pyg data\n')

    print("##########################################")
    print("### Complete generating PyG graph of " + ("session2" if "session2" in args.storefile else "session1"))
    print("##########################################")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', type=str, default=None)  # icdm2022_session1_edges.csv
    parser.add_argument('--node', type=str, default=None)  # icdm2022_session1_nodes.csv
    parser.add_argument('--label', type=str, default=None)  # icdm2022_session1_train_labels.csv
    parser.add_argument('--storefile', type=str, default=None)
    parser.add_argument('--reload', type=bool, default=False, help="Whether node features should be reloaded")
    args = parser.parse_args()
    if "session2" in args.storefile:
        edge_size = 120691444
        node_size = 10284026
    else:
        edge_size = 157814864
        node_size = 13806619
    if args.graph is not None and args.storefile is not None and args.node is not None:
        format_pyg_graph(args.graph, args.node, args.storefile, args.label)
        # read_node_atts(args.node, args.storefile, args.label)
