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

# inits
print('-----Loading -----')
os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
hgraph = torch.load(args.dataset)
labeled_class = args.labeled_class
num_relations = len(hgraph.edge_types)
flag = 0
homo = ['RGCN', 'FASTRGCN', 'GCN', 'ChebGCN', 'SAGEGCN', 'GraphGCN', 'GatedGraphGCN', 'GAT', 'GATv2', 'Transformer',
        'TAG', 'ARMA', 'SG', 'MF']

if args.model in homo:
    flag = 0
elif args.model == 'HGT':
    flag = 1
    hgraph.to(device)

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
    train_loader = NeighborLoader(hgraph, input_nodes=(labeled_class, train_idx),
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

if args.inference:
    model = torch.load(osp.join('best_model', args.model + ".pth"))
else:
    model = {
        'RGCN': lambda: RGCN(in_channels=args.in_dim, hidden_channels=args.h_dim, out_channels=2,
                             num_relations=num_relations, n_layers=args.n_layers),
        'HGT': lambda: HGT(hidden_channels=args.h_dim, out_channels=2, num_heads=8, num_layers=args.n_layers,
                           hgraph=hgraph, labeled_class=labeled_class),
        'FASTRGCN': lambda: FASTRGCN(in_channels=args.in_dim, hidden_channels=args.h_dim, out_channels=2,
                                     num_relations=num_relations, n_layers=args.n_layers),
        'GCN': lambda: GCN(in_channels=args.in_dim, hidden_channels=args.h_dim, out_channels=2,
                           num_relations=num_relations, n_layers=args.n_layers),
        'ChebGCN': lambda: ChebGCN(in_channels=args.in_dim, hidden_channels=args.h_dim, out_channels=2,
                                   num_relations=num_relations, n_layers=args.n_layers),
        'SAGEGCN': lambda: SAGEGCN(in_channels=args.in_dim, hidden_channels=args.h_dim, out_channels=2,
                                   num_relations=num_relations, n_layers=args.n_layers),
        'GraphGCN': lambda: GraphGCN(in_channels=args.in_dim, hidden_channels=args.h_dim, out_channels=2,
                                     num_relations=num_relations, n_layers=args.n_layers),
        'GatedGraphGCN': lambda: GatedGraphGCN(in_channels=args.in_dim, hidden_channels=args.h_dim, out_channels=2,
                                               num_relations=num_relations, n_layers=args.n_layers),
        'GAT': lambda: GAT(in_channels=args.in_dim, hidden_channels=args.h_dim, out_channels=2, heads=8),
        'GATv2': lambda: GATv2(in_channels=args.in_dim, hidden_channels=args.h_dim, out_channels=2, heads=8),
        'Transformer': lambda: Transformer(in_channels=args.in_dim, hidden_channels=args.h_dim, out_channels=2,
                                           num_relations=num_relations, n_layers=args.n_layers),
        'TAG': lambda: TAG(in_channels=args.in_dim, hidden_channels=args.h_dim, out_channels=2,
                           num_relations=num_relations, n_layers=args.n_layers),
        'ARMA': lambda: ARMA(in_channels=args.in_dim, hidden_channels=args.h_dim, out_channels=2,
                             num_relations=num_relations, n_layers=args.n_layers),
        'SG': lambda: SG(in_channels=args.in_dim, hidden_channels=args.h_dim, out_channels=2,
                         num_relations=num_relations, n_layers=args.n_layers),
        'MF': lambda: MF(in_channels=args.in_dim, hidden_channels=args.h_dim, out_channels=2,
                         num_relations=num_relations, n_layers=args.n_layers),
    }[args.model]()
    model.to(device)

    optimizer = {
        'Adam': lambda: torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay),
    }[args.optimizer]()


def train(epoch):
    model.train()

    pbar = tqdm(total=int(len(train_loader.dataset)), ascii=True)
    pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = total_correct = total_examples = 0
    y_pred = []
    y_true = []
    for batch in train_loader:
        optimizer.zero_grad()
        batch_size = batch[labeled_class].batch_size
        y = batch[labeled_class].y[:batch_size].to(device)

        if flag == 0:
            start = 0
            for ntype in batch.node_types:
                if ntype == labeled_class:
                    break
                start += batch[ntype].num_nodes

            batch = batch.to_homogeneous()
            y_hat = model(batch.x.to(device), batch.edge_index.to(device), batch.edge_type.to(device))[
                    start:start + batch_size]
        else:
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
        batch_size = batch[labeled_class].batch_size
        y = batch[labeled_class].y[:batch_size].to(device)
        if flag == 0:
            start = 0
            for ntype in batch.node_types:
                if ntype == labeled_class:
                    break
                start += batch[ntype].num_nodes

            batch = batch.to_homogeneous()

            y_hat = model(batch.x.to(device), batch.edge_index.to(device), batch.edge_type.to(device))[
                    start:start + batch_size]
        else:
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
        else:
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
                torch.save(model, args.store_path.format(args.model))
                best_ap = val_ap
            print(
                f'Val: Epoch: {epoch:02d}, Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, AP_Score: {val_ap:.4f}, Best AP: {best_ap: .4f}')
            val_ap_list.append(float(val_ap))
            ave_val_ap = np.average(val_ap_list)
            end = epoch
    print(f"Complete Trianing (Model: {args.model})")

if args.inference:
    y_pred = test()
    with open(args.json_file, 'w+') as f:
        for i in range(len(test_id)):
            y_dict = {"item_id": int(test_id[i]), "score": float(y_pred[i])}
            json.dump(y_dict, f)
            f.write('\n')
