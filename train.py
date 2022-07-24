import os

import torch

from args import args

os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
hgraph = torch.load(args.dataset)
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
