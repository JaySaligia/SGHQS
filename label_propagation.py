import os
import os.path as osp
import torch
import json

from config import args
from model import *
from tqdm import tqdm
from torch_geometric.loader import NeighborLoader
from sklearn.metrics import average_precision_score
import numpy as np
from gtrick.pyg import LabelPropagation


# inits
print('-----Loading -----')
os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
hgraph = torch.load(args.dataset)
labeled_class = args.labeled_class
num_relations = len(hgraph.edge_types)
flag = 0
homo = ['RGCN', 'FASTRGCN', 'GCN', 'ChebGCN', 'SAGEGCN', 'GraphGCN', 'GatedGraphGCN', 'GAT', 'GATv2', 'Transformer',
        'TAG', 'ARMA', 'SG', 'MF', 'EG']
peculiar = ['superGAT']

if args.model in homo:
    flag = 0
elif args.model == 'HGT':
    flag = 1
    hgraph.to(device)
elif args.model in peculiar:
    flag = 2


train_idx = hgraph[labeled_class].pop('train_idx')

test_id = [int(x) for x in open(args.test_file).readlines()]
converted_test_id = []
for i in test_id:
    converted_test_id.append(hgraph['item'].maps[i])
test_idx = torch.LongTensor(converted_test_id)

# print(len(hgraph[labeled_class].y))
# print(len(hgraph.to_homogeneous().edge_index))
# Mini-Batch
if args.inference:
    model = torch.load(osp.join('best_model', args.model + ".pth"))
    print('Load {} model'.format(args.model))

if args.inference:
    test_loader = NeighborLoader(hgraph, input_nodes=(labeled_class, test_idx),
                             num_neighbors={key: [args.fanout] * args.n_layers for key in hgraph.edge_types},
                             shuffle=False, batch_size=args.batch_size)

@torch.no_grad()
def test():
    model.eval()
    pbar = tqdm(total=int(len(test_loader.dataset)), ascii=True)
    pbar.set_description(f'Generate Final Result:')
    y_pred = []
    count = 0
    for batch in test_loader:
        count += 1
        if count > 1:
            break
        batch_size = batch[labeled_class].batch_size
        if flag == 0:
            start = 0
            for ntype in batch.node_types:
                if ntype == labeled_class:
                    break
                start += batch[ntype].num_nodes

            batch = batch.to_homogeneous()
            y_hat = model(batch.x.to(device), batch.edge_index.to(device), batch.edge_type.to(device))[
                    start:start + batch_size]
        elif flag == 1:
            y_hat = model(batch.x_dict, batch.edge_index_dict)[0:batch_size]
        elif flag == 2:
            start = 0
            for ntype in batch.node_types:
                if ntype == labeled_class:
                    break
                start += batch[ntype].num_nodes

            batch = batch.to_homogeneous()
            y_hat = model(batch.x.to(device), batch.edge_index.to(device))[0][start:start + batch_size]
        pbar.update(batch_size)
        y_pred.append(F.softmax(y_hat, dim=1)[:, 1].detach().cpu())
    pbar.close()

    return torch.hstack(y_pred)


y = hgraph[labeled_class].y + 1
adj_t = hgraph.to_homogeneous().edge_index

if args.inference:
    y_pred = test()
    if args.label_propagation:
        lp = LabelPropagation(args.lp_layers, args.lp_alpha)
        yh = lp(y, adj_t)
        y_pred = torch.argmax(y_pred + yh, dim=-1, keepdim=True)
    with open(args.json_file, 'w+') as f:
        for i in range(len(test_id)):
            y_dict = {"item_id": int(test_id[i]), "score": float(y_pred[i])}
            json.dump(y_dict, f)
            f.write('\n')
