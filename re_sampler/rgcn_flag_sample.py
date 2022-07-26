# 将重采样与flag技巧合并
import os
import os.path as osp
import argparse
import json

import numpy as np
import torch
import random
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

from torch_geometric.nn import RGCNConv
from sklearn.metrics import average_precision_score
from pick_sample import pick_step, get_sample_graph
from gtrick import FLAG

parser = argparse.ArgumentParser()
# parser.add_argument('--dataset', type=str, default='/data/xuwenjie/Ali/SGHQS/dataset/pyg_data/icdm2022_session1.pt')
parser.add_argument('--dataset', type=str, default='/data/liuben/icdm_dataset/session2/icdm2022_session2.pt')
parser.add_argument('--store_path', type=str, default='/data/xuwenjie/Ali/SGHQS/best_model/{}.pth')
parser.add_argument('--labeled-class', type=str, default='item')
parser.add_argument("--batch-size", type=int, default=200,
                    help="Mini-batch size. If -1, use full graph training.")
parser.add_argument("--fanout", type=int, default=150,
                    help="Fan-out of neighbor sampling.")
parser.add_argument("--n-layers", type=int, default=2,
                    help="number of propagation rounds")
parser.add_argument("--h-dim", type=int, default=256,
                    help="number of hidden units")
parser.add_argument("--in-dim", type=int, default=256,
                    help="number of hidden units")
parser.add_argument("--n-bases", type=int, default=8,
                    help="number of filter weight matrices, default: -1 [use all]")
parser.add_argument("--validation", type=bool, default=True)
parser.add_argument("--early_stopping", type=int, default=6)
parser.add_argument("--n-epoch", type=int, default=100)
# parser.add_argument("--test-file", type=str, default="/home/icdm/icdm2022_large/test_session1_ids.csv")
parser.add_argument("--test-file", type=str, default="/data/liuben/icdm_dataset/session2/icdm2022_session2_test_ids.txt")

parser.add_argument("--json-file", type=str, default="pyg_pred_2.json")
parser.add_argument("--inference", type=bool, default=False)
# parser.add_argument("--record-file", type=str, default="record.txt")
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--model-id", type=str, default="rgcn_flag_sample")
parser.add_argument("--device-id", type=str, default="2")

args = parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


set_seed(4399)

os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
hgraph = torch.load(args.dataset)
# taget ndoe adjList
item_adj = [hgraph.edge_stores[i].edge_index for i in [0, 1, 9, 10]]

labeled_class = args.labeled_class

if not args.inference:
    train_idx = hgraph[labeled_class].pop('train_idx')
    if args.validation:
        val_idx = hgraph[labeled_class].pop('val_idx')
else:
    test_id = [int(x) for x in open(args.test_file).readlines()]
    converted_test_id = []
    for i in test_id:
        converted_test_id.append(hgraph['item'].maps[i])
    test_idx = torch.LongTensor(converted_test_id)

# Mini-Batch
if not args.inference:
    train_label = hgraph['item'].y[train_idx]

# # No need to maintain these features during evaluation:
# # Add global node index information.
# test_loader.data.num_nodes = data.num_nodes
# test_loader.data.n_id = torch.arange(data.num_nodes)


num_relations = len(hgraph.edge_types)


class RGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n_layers=2):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.relu = F.relu
        self.convs.append(RGCNConv(in_channels, hidden_channels, num_relations, num_bases=args.n_bases))
        for i in range(n_layers - 2):
            self.convs.append(RGCNConv(hidden_channels, hidden_channels, num_relations, num_bases=args.n_bases))
        self.convs.append(RGCNConv(hidden_channels, out_channels, num_relations, num_bases=args.n_bases))

    def forward(self, x, edge_index, edge_type, perturb=None):
        if perturb is not None:
            x = x + perturb
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_type)
            if i < len(self.convs) - 1:
                x = x.relu_()
                x = F.dropout(x, p=0.4, training=self.training)
        return x


if args.inference:
    # model = torch.load(osp.join('best_model', args.model_id + ".pth"))
    model = torch.load(args.store_path.format(args.model_id))
else:
    model = RGCN(in_channels=args.in_dim, hidden_channels=args.h_dim, out_channels=2, n_layers=args.n_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


def train(epoch, flag):
    model.train()
    total_loss = total_correct = total_examples = 0
    y_pred = []
    y_true = []
    sampled_idx_train = pick_step(train_idx, train_label, item_adj, size=train_label.sum()*2)
    random.shuffle(sampled_idx_train)
    num_batches = int(len(sampled_idx_train) / args.batch_size) + 1

    pbar = tqdm(total=num_batches, ascii=True)
    pbar.set_description(f'Epoch {epoch:02d}')

    for batch in range(num_batches):
        optimizer.zero_grad()
        i_start = batch * args.batch_size
        i_end = min((batch + 1) * args.batch_size, len(sampled_idx_train))
        batch_nodes = sampled_idx_train[i_start:i_end]
        batch_labels = hgraph[labeled_class].y[batch_nodes].to(device)
        batch_graph = get_sample_graph(batch_nodes, hgraph)
        start = 0
        for ntype in batch_graph.node_types:
            if ntype == labeled_class:
                break
            start += batch_graph[ntype].num_nodes

        batch_graph = batch_graph.to_homogeneous()

        y_hat = lambda perturb: model(batch_graph.x.to(device), batch_graph.edge_index.to(device), batch_graph.edge_type.to(device), perturb)[
                start:start + len(batch_nodes)]

        # y_hat = model(batch_graph.x.to(device), batch_graph.edge_index.to(device),
        #               batch_graph.edge_type.to(device))[start:start + len(batch_nodes)]
        # loss = F.cross_entropy(y_hat, batch_labels)
        loss, out = flag(model, y_hat, batch_graph.x.shape[0], batch_labels)
        loss = loss.item()
        # loss.backward()
        # optimizer.step()
        y_pred.append(F.softmax(out, dim=1)[:, 1].detach().cpu())
        y_true.append(batch_labels.cpu())
        total_loss += float(loss) * len(batch_nodes)
        total_correct += int((out.argmax(dim=-1) == batch_labels).sum())
        total_examples += len(batch_nodes)
        pbar.update(1)
    pbar.close()
    ap_score = average_precision_score(torch.hstack(y_true).numpy(), torch.hstack(y_pred).numpy())

    return total_loss / total_examples, total_correct / total_examples, ap_score


@torch.no_grad()
def val():
    model.eval()
    total_loss = total_correct = total_examples = 0
    y_pred = []
    y_true = []
    num_batches = int(len(val_idx) / args.batch_size) + 1

    pbar = tqdm(total=num_batches, ascii=True)
    pbar.set_description(f'Epoch {epoch:02d}')
    for batch in range(num_batches):
        i_start = batch * args.batch_size
        i_end = min((batch + 1) * args.batch_size, len(val_idx))
        batch_nodes = val_idx[i_start:i_end]
        batch_labels = hgraph[labeled_class].y[batch_nodes].to(device)
        batch_graph = get_sample_graph(batch_nodes, hgraph)

        start = 0
        for ntype in batch_graph.node_types:
            if ntype == labeled_class:
                break
            start += batch_graph[ntype].num_nodes

        batch_graph = batch_graph.to_homogeneous()

        y_hat = model(batch_graph.x.to(device), batch_graph.edge_index.to(device),
                      batch_graph.edge_type.to(device))[start:start + len(batch_nodes)]
        loss = F.cross_entropy(y_hat, batch_labels)
        y_pred.append(F.softmax(y_hat, dim=1)[:, 1].detach().cpu())
        y_true.append(batch_labels.cpu())
        total_loss += float(loss) * len(batch_nodes)
        total_correct += int((y_hat.argmax(dim=-1) == batch_labels).sum())
        total_examples += len(batch_nodes)
        pbar.update(1)
    pbar.close()
    ap_score = average_precision_score(torch.hstack(y_true).numpy(), torch.hstack(y_pred).numpy())

    return total_loss / total_examples, total_correct / total_examples, ap_score


@torch.no_grad()
def test():
    model.eval()
    y_pred = []
    num_batches = int(len(test_idx) / args.batch_size) + 1

    pbar = tqdm(total=num_batches, ascii=True)
    pbar.set_description('Generate Final Result:')
    for batch in range(num_batches):
        i_start = batch * args.batch_size
        i_end = min((batch + 1) * args.batch_size, len(test_idx))
        batch_nodes = test_idx[i_start:i_end]
        batch_graph = get_sample_graph(batch_nodes, hgraph)

        start = 0
        for ntype in batch_graph.node_types:
            if ntype == labeled_class:
                break
            start += batch_graph[ntype].num_nodes

        batch_graph = batch_graph.to_homogeneous()

        y_hat = model(batch_graph.x.to(device), batch_graph.edge_index.to(device),
                      batch_graph.edge_type.to(device))[start:start + len(batch_nodes)]
        pbar.update(1)
        y_pred.append(F.softmax(y_hat, dim=1)[:, 1].detach().cpu())
    pbar.close()

    return torch.hstack(y_pred)


if not args.inference:
    print("Start training")
    val_ap_list = []
    ave_val_ap = 0
    end = 0
    loss_func = nn.CrossEntropyLoss()
    best_ap = 0
    flag = FLAG(args.in_dim, loss_func, optimizer)
    for epoch in range(1, args.n_epoch + 1):
        train_loss, train_acc, train_ap = train(epoch, flag)
        print(f'Train: Epoch {epoch:02d}, Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, AP_Score: {train_ap:.4f}')
        if args.validation and epoch >= args.early_stopping:
            val_loss, val_acc, val_ap = val()
            print(f'Val: Epoch: {epoch:02d}, Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, AP_Score: {val_ap:.4f}')
            if val_ap <= ave_val_ap:
                print("Early Stopping")
                break
            if val_ap > best_ap:
                torch.save(model, args.store_path.format(args.model_id))
                best_ap = val_ap
            val_ap_list.append(float(val_ap))
            ave_val_ap = np.average(val_ap_list)
            end = epoch
    print(f"Complete Trianing (Model id: {args.model_id})")

#    with open(args.record_file, 'a+') as f:
#        f.write(f"{args.model_id:2d} {args.h_dim:3d} {args.n_layers:2d} {args.lr:.4f} {end:02d} {float(val_ap_list[-1]):.4f} {np.argmax(val_ap_list)+5:02d} {float(np.max(val_ap_list)):.4f}\n")


if args.inference:
    y_pred = test()
    with open(args.json_file, 'w+') as f:
        for i in range(len(test_id)):
            y_dict = {}
            y_dict["item_id"] = int(test_id[i])
            y_dict["score"] = float(y_pred[i])
            json.dump(y_dict, f)
            f.write('\n')
