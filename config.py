import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='./dataset/pyg_data/icdm2022_session1.pt')
parser.add_argument('--store_path', type=str, default='./best_model/{}.pthls')
parser.add_argument('--labeled-class', type=str, default='item')
parser.add_argument('--batch-size', type=int, default=128,
                    help='Mini-batch size. If -1, use full graph training.')
parser.add_argument('--fanout', type=int, default=-1,
                    help='Fan-out of neighbor sampling.')
parser.add_argument('--n-layers', type=int, default=2,
                    help='number of propagation rounds')
parser.add_argument('--h-dim', type=int, default=16,
                    help='number of hidden units')
parser.add_argument('--in-dim', type=int, default=256,
                    help='number of hidden units')
parser.add_argument('--n-bases', type=int, default=8,
                    help='number of filter weight matrices, default: -1 [use all]')
parser.add_argument('--validation', type=bool, default=False)
parser.add_argument('--early_stopping', type=int, default=10)
parser.add_argument('--n-epoch', type=int, default=1)
parser.add_argument('--test-file', type=str, default='/home/icdm/icdm2022_large/test_session1_ids.csv')
parser.add_argument('--json-file', type=str, default='pyg_pred.json')
parser.add_argument('--inference', type=bool, default=False)
# parser.add_argument('--record-file', type=str, default='record.txt')
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--model-id', type=str, default='0')
parser.add_argument('--device-id', type=str, default='0')

parser.add_argument('--model', type=str, default='RGCN')
parser.add_argument('--optimizer', type=str, default='Adam')
args = parser.parse_args()
