import os
import os.path as osp
import argparse
from tqdm import tqdm
import json
import numpy as np
import random

import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader, HGTLoader
from torch_geometric.datasets import DBLP

from torch_geometric.nn import HeteroConv, Linear, SAGEConv, GATConv, ResGatedGraphConv
from sklearn.metrics import average_precision_score


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='/data/pengmiao/ICDM_dataset/pyg_data/icdm2022_session1.pt')
parser.add_argument('--labeled-class', type=str, default='item')
parser.add_argument("--batch-size", type=int, default=256,
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
parser.add_argument("--early_stopping", type=int, default=10)
parser.add_argument("--n-epoch", type=int, default=100)
parser.add_argument("--test-file", type=str, default="/data/pengmiao/ICDM_dataset/icdm2022_session1_test_ids.txt")
parser.add_argument("--json-file", type=str, default="pyg_pred.json")
parser.add_argument("--inference", type=bool, default=False)
# parser.add_argument("--record-file", type=str, default="record.txt")
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--model-id", type=str, default="hg_metapath_1")
parser.add_argument("--device-id", type=str, default="9")
parser.add_argument("--record_file", type=str, default="/data/pengmiao/workplace/pycharm/icdm_graph_competition/pyHGT/parameter_record.txt")
parser.add_argument('--use_hgt_loader', action='store_true')

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

setup_seed(12138)

# os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id  # args.device_id
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(args.device_id)
hgraph = torch.load(args.dataset).to(device, 'x', 'y')  # .to(device)
meta = hgraph.metadata()

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
if not args.use_hgt_loader:
    if args.inference == False:
        train_loader = NeighborLoader(hgraph, input_nodes=(labeled_class, train_idx),  # input_nodes: train_mask
                                      num_neighbors=[args.fanout] * args.n_layers,
                                      shuffle=True, batch_size=args.batch_size)

        if args.validation:
            val_loader = NeighborLoader(hgraph, input_nodes=(labeled_class, val_idx),
                                        num_neighbors=[args.fanout] * args.n_layers,
                                        shuffle=False, batch_size=args.batch_size)
    else:
        test_loader = NeighborLoader(hgraph, input_nodes=(labeled_class, test_idx),
                                     num_neighbors=[args.fanout] * args.n_layers,
                                     shuffle=False, batch_size=args.batch_size)
else:
    print("Using HGTLoader...............")
    kwargs = {'batch_size': args.batch_size, 'num_workers': 6, 'persistent_workers': True}
    if args.inference == False:
        train_loader = HGTLoader(hgraph, num_samples=[args.fanout] * 4, shuffle=True,
                                 input_nodes=(labeled_class, train_idx), **kwargs)

        if args.validation:
            val_loader = HGTLoader(hgraph, num_samples=[args.fanout] * 4,
                                   input_nodes=(labeled_class, val_idx), **kwargs)
    else:
        test_loader = HGTLoader(hgraph, num_samples=[args.fanout] * 4,
                                   input_nodes=(labeled_class, test_idx), **kwargs)

# # No need to maintain these features during evaluation:
# # Add global node index information.
# test_loader.data.num_nodes = data.num_nodes
# test_loader.data.n_id = torch.arange(data.num_nodes)


num_relations = len(hgraph.edge_types)


class HeteroGNN(torch.nn.Module):
    def __init__(self, metadata, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                edge_type: ResGatedGraphConv((-1, -1), hidden_channels)  # SAGEConv((-1, -1), hidden_channels)
                for edge_type in metadata[1]
            }, aggr='max')  # TODO: Max比sum效果好一些
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}
        return self.lin(x_dict[labeled_class])


if args.inference:
    model = torch.load(osp.join('best_model', args.model_id + ".pth"))
else:
    model = HeteroGNN(hgraph.metadata(), hidden_channels=args.h_dim, out_channels=2, num_layers=args.n_layers).to(device)
    # with torch.no_grad():  # Initialize lazy modules.
    #     hgraph = hgraph.to(device, 'edge_index')
    #     out = model(hgraph.x_dict, hgraph.edge_index_dict)
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

        batch = batch.to(device, 'edge_index')

        batch_size = batch[labeled_class].batch_size
        y = batch[labeled_class].y[:batch_size].to(device)

        out = model(batch.x_dict, batch.edge_index_dict)[0:batch_size]
        y_hat = out  # [labeled_class]
        loss = F.cross_entropy(y_hat, y)
        loss.backward()
        optimizer.step()
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
def val():
    model.eval()
    pbar = tqdm(total=int(len(val_loader.dataset)), ascii=True)
    pbar.set_description(f'Epoch {epoch:02d}')
    total_loss = total_correct = total_examples = 0
    y_pred = []
    y_true = []
    for batch in val_loader:
        batch = batch.to(device, 'edge_index')

        batch_size = batch[labeled_class].batch_size
        y = batch[labeled_class].y[:batch_size].to(device)

        y_hat = model(batch.x_dict, batch.edge_index_dict)[0:batch_size]
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
        batch = batch.to(device, 'edge_index')

        batch_size = batch[labeled_class].batch_size

        y_hat = model(batch.x_dict, batch.edge_index_dict)[0:batch_size]
        pbar.update(batch_size)
        y_pred.append(F.softmax(y_hat, dim=1)[:, 1].detach().cpu())
    pbar.close()

    return torch.hstack(y_pred)


if not args.inference:
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
            if val_ap <= ave_val_ap * 0.95:
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
    print(f"Complete Trianing (Model: {args.model})")

if args.inference == True:
    y_pred = test()
    with open(args.json_file, 'w+') as f:
        for i in range(len(test_id)):
            y_dict = {}
            y_dict["item_id"] = int(test_id[i])
            y_dict["score"] = float(y_pred[i])
            json.dump(y_dict, f)
            f.write('\n')
