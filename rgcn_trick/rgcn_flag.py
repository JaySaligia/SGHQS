import os
import os.path as osp
import argparse
import json

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from tqdm import tqdm
import random

from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import RGCNConv
from sklearn.metrics import average_precision_score
from gtrick import FLAG

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='/data/pengmiao/ICDM_dataset/pyg_data/icdm2022_session1.pt')
parser.add_argument('--labeled-class', type=str, default='item')
parser.add_argument("--batch-size", type=int, default=200,
                    help="Mini-batch size. If -1, use full graph training.")
parser.add_argument("--fanout", type=int, default=-1,
                    help="Fan-out of neighbor sampling.")
parser.add_argument("--n-layers", type=int, default=2,
                    help="number of propagation rounds")
parser.add_argument("--h-dim", type=int, default=16,
                    help="number of hidden units")
parser.add_argument("--in-dim", type=int, default=256,
                    help="number of hidden units")
parser.add_argument("--n-bases", type=int, default=8,
                    help="number of filter weight matrices, default: -1 [use all]")
parser.add_argument("--validation", type=bool, default=False)
parser.add_argument("--early_stopping", type=int, default=6)
parser.add_argument("--n-epoch", type=int, default=1)
parser.add_argument("--test-file", type=str, default="/data/pengmiao/ICDM_dataset/icdm2022_session1_test_ids.txt")
parser.add_argument("--json-file", type=str, default="pyg_pred.json")
parser.add_argument("--inference", type=bool, default=False)
# parser.add_argument("--record-file", type=str, default="record.txt")
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--model-id", type=str, default="rgcn_flag_2")
parser.add_argument("--device-id", type=str, default="5")

args = parser.parse_args()

def setup_seed(seed):
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

setup_seed(4399)

os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
hgraph = torch.load(args.dataset)

# use_gdc
# print("Using GDC...")
# transform = T.GDC(
#     self_loop_weight=1,
#     normalization_in='sym',
#     normalization_out='col',
#     diffusion_kwargs=dict(method='ppr', alpha=0.05),
#     sparsification_kwargs=dict(method='topk', k=128, dim=0),
#     exact=True,
# )

labeled_class = args.labeled_class

if args.inference == False:
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
if args.inference == False:
    train_loader = NeighborLoader(hgraph, input_nodes=(labeled_class, train_idx),
                                  num_neighbors={key: [args.fanout] * args.n_layers for key in hgraph.edge_types},
                                  shuffle=True, batch_size=args.batch_size)

    if args.validation:
        val_loader = NeighborLoader(hgraph, input_nodes=(labeled_class, val_idx),
                                    num_neighbors={key: [args.fanout] * args.n_layers for key in hgraph.edge_types},
                                    shuffle=False, batch_size=args.batch_size)
else:
    test_loader = NeighborLoader(hgraph, input_nodes=(labeled_class, test_idx),
                                 num_neighbors={key: [args.fanout] * args.n_layers for key in hgraph.edge_types},
                                 shuffle=False, batch_size=args.batch_size)

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

    # add a param perturb to pass perturb
    def forward(self, x, edge_index, edge_type, perturb=None):
        # add perturb to x, note that do not use x += perturb
        if perturb is not None:
            x = x + perturb

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_type)
            if i < len(self.convs) - 1:
                x = x.relu_()
                x = F.dropout(x, p=0.4, training=self.training)
        return x


if args.inference:
    model = torch.load(osp.join('best_model', args.model_id + ".pth"))
else:
    model = RGCN(in_channels=args.in_dim, hidden_channels=args.h_dim, out_channels=2, n_layers=args.n_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


def train(epoch):
    model.train()

    pbar = tqdm(total=int(len(train_loader.dataset)), ascii=True)
    pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = total_correct = total_examples = 0
    y_pred = []
    y_true = []
    for batch in train_loader:
        optimizer.zero_grad()
        loss_func = nn.CrossEntropyLoss()
        # define flag, params: in_feats, loss_func, optimizer
        flag = FLAG(args.in_dim, loss_func, optimizer)

        batch_size = batch[labeled_class].batch_size
        y = batch[labeled_class].y[:batch_size].to(device)

        start = 0
        for ntype in batch.node_types:
            if ntype == labeled_class:
                break
            start += batch[ntype].num_nodes

        batch = batch.to_homogeneous()
        # batch = transform(batch)

        # define a forward func to get the output of the model
        y_hat = lambda perturb: model(batch.x.to(device), batch.edge_index.to(device), batch.edge_type.to(device), perturb)[
                start:start + batch_size]

        # run flag to get loss and output
        loss, out = flag(model, y_hat, batch.x.shape[0], y)
        loss = loss.item()
        # loss = F.cross_entropy(y_hat, y)
        # loss.backward()
        # optimizer.step()
        y_pred.append(F.softmax(out, dim=1)[:, 1].detach().cpu())
        y_true.append(y.cpu())
        total_loss += float(loss) * batch_size
        total_correct += int((out.argmax(dim=-1) == y).sum())
        total_examples += batch_size
        pbar.update(batch_size)
    pbar.close()
    ap_score = average_precision_score(torch.hstack(y_true).numpy(), torch.hstack(y_pred).numpy())

    return total_loss / total_examples, total_correct / total_examples, ap_score


@torch.no_grad()
def val():
    model.eval()
    pbar = tqdm(total=int(len(val_loader.dataset)), ascii=True)
    pbar.set_description(f'Epoch {epoch:02d}')
    total_loss = total_correct = total_examples = 0
    y_pred = []
    y_true = []
    for batch in val_loader:
        batch_size = batch[labeled_class].batch_size
        y = batch[labeled_class].y[:batch_size].to(device)
        start = 0
        for ntype in batch.node_types:
            if ntype == labeled_class:
                break
            start += batch[ntype].num_nodes

        batch = batch.to_homogeneous()

        y_hat = model(batch.x.to(device), batch.edge_index.to(device), batch.edge_type.to(device))[
                start:start + batch_size]
        loss = F.cross_entropy(y_hat, y)
        y_pred.append(F.softmax(y_hat, dim=1)[:, 1].detach().cpu())
        y_true.append(y.cpu())
        total_loss += float(loss) * batch_size
        total_correct += int((y_hat.argmax(dim=-1) == y).sum())
        total_examples += batch_size
        pbar.update(batch_size)
    pbar.close()
    ap_score = average_precision_score(torch.hstack(y_true).numpy(), torch.hstack(y_pred).numpy())

    return total_loss / total_examples, total_correct / total_examples, ap_score


@torch.no_grad()
def test():
    model.eval()
    pbar = tqdm(total=int(len(test_loader.dataset)), ascii=True)
    pbar.set_description(f'Generate Final Result:')
    y_pred = []
    for batch in test_loader:
        batch_size = batch[labeled_class].batch_size
        start = 0
        for ntype in batch.node_types:
            if ntype == labeled_class:
                break
            start += batch[ntype].num_nodes

        batch = batch.to_homogeneous()
        y_hat = model(batch.x.to(device), batch.edge_index.to(device), batch.edge_type.to(device))[
                start:start + batch_size]
        pbar.update(batch_size)
        y_pred.append(F.softmax(y_hat, dim=1)[:, 1].detach().cpu())
    pbar.close()

    return torch.hstack(y_pred)


if args.inference == False:
    print("Start training")
    val_ap_list = []
    ave_val_ap = 0
    end = 0
    best_ap = 0
    for epoch in range(1, args.n_epoch + 1):
        train_loss, train_acc, train_ap = train(epoch)
        print(f'Train: Epoch {epoch:02d}, Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, AP_Score: {train_ap:.4f}')
        if args.validation and epoch >= args.early_stopping:
            val_loss, val_acc, val_ap = val()
            if val_ap <= ave_val_ap:
                print(
                    f'Val: Epoch: {epoch:02d}, Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, AP_Score: {val_ap:.4f}, Best AP: {best_ap: .4f}')
                print("Early Stopping")
                break
            if val_ap > best_ap:
                torch.save(model, osp.join("best_model/", args.model_id + ".pth"))
                best_ap = val_ap
            print(
                f'Val: Epoch: {epoch:02d}, Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, AP_Score: {val_ap:.4f}, Best AP: {best_ap: .4f}')
            val_ap_list.append(float(val_ap))
            ave_val_ap = np.average(val_ap_list)
            end = epoch
    print(f"Complete Trianing (Model id: {args.model_id})")

#    with open(args.record_file, 'a+') as f:
#        f.write(f"{args.model_id:2d} {args.h_dim:3d} {args.n_layers:2d} {args.lr:.4f} {end:02d} {float(val_ap_list[-1]):.4f} {np.argmax(val_ap_list)+5:02d} {float(np.max(val_ap_list)):.4f}\n")


if args.inference == True:
    y_pred = test()
    with open(args.json_file, 'w+') as f:
        for i in range(len(test_id)):
            y_dict = {}
            y_dict["item_id"] = int(test_id[i])
            y_dict["score"] = float(y_pred[i])
            json.dump(y_dict, f)
            f.write('\n')



