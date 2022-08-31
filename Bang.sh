if [ $1 = "RGCN" ]
then echo "python train.py --dataset './dataset/pyg_data/icdm2022_session1.pt' --h-dim 256 --n-bases 8 --n-layers 3 --fanout 150 --n-epoch 13 --early_stopping 4 --validation True --lr 0.001 --batch-size 200 --model-id 'RGCN' --model 'RGCN' --optimizer 'Adam' --device-id '0'"
elif [ $1 = "HGT" ]
then echo "python train.py --dataset './dataset/pyg_data/icdm2022_session1.pt' --h-dim 512 --n-bases 8 --n-layers 3 --fanout 150 --n-epoch 100 --early_stopping 10 --validation True --lr 0.001 --batch-size 512 --model-id 'hgt_4' --model 'HGT' --optimizer 'Adam' --device-id 6"
elif [ $1 = "FASTRGCN" ]
then echo "python train.py --dataset './dataset/pyg_data/icdm2022_session1.pt' --h-dim 256 --n-bases 8 --n-layers 3 --fanout 50 --n-epoch 100 --early_stopping 10 --validation True --lr 0.01 --batch-size 256 --model-id 'FASTRGCN' --model 'FASTRGCN' --optimizer 'Adam' --device-id 6"
elif [ $1 = "GCN" ]
then echo "python train.py --dataset './dataset/pyg_data/icdm2022_session1.pt' --h-dim 256 --n-bases 8 --n-layers 3 --fanout 50 --n-epoch 100 --early_stopping 10 --validation True --lr 0.01 --batch-size 256 --model-id 'GCN' --model 'GCN' --optimizer 'Adam' --device-id 6"
elif [ $1 = "ChebGCN" ]
then echo "python train.py --dataset './dataset/pyg_data/icdm2022_session1.pt' --h-dim 256 --n-bases 8 --n-layers 3 --fanout 50 --n-epoch 100 --early_stopping 10 --validation True --lr 0.01 --batch-size 256 --model-id 'ChebGCN' --model 'ChebGCN' --optimizer 'Adam' --device-id 6"
elif [ $1 = "SAGEGCN" ]
then echo "python train.py --dataset './dataset/pyg_data/icdm2022_session1.pt' --h-dim 256 --n-bases 8 --n-layers 3 --fanout 50 --n-epoch 100 --early_stopping 10 --validation True --lr 0.01 --batch-size 256 --model-id 'SAGEGCN' --model 'SAGEGCN' --optimizer 'Adam' --device-id 6"
elif [ $1 = "GraphGCN" ]
then echo "python train.py --dataset './dataset/pyg_data/icdm2022_session1.pt' --h-dim 256 --n-bases 8 --n-layers 3 --fanout 50 --n-epoch 100 --early_stopping 10 --validation True --lr 0.01 --batch-size 256 --model-id 'GraphGCN' --model 'GraphGCN' --optimizer 'Adam' --device-id 6"
elif [ $1 = "GatedGraphGCN" ]
then echo "python train.py --dataset './dataset/pyg_data/icdm2022_session1.pt' --h-dim 256 --n-bases 8 --n-layers 3 --fanout 50 --n-epoch 100 --early_stopping 10 --validation True --lr 0.01 --batch-size 256 --model-id 'GatedGraphGCN' --model 'GatedGraphGCN' --optimizer 'Adam' --device-id 6"
elif [ $1 = "GAT" ]
then echo "python train.py --dataset './dataset/pyg_data/icdm2022_session1.pt' --h-dim 8 --n-bases 8 --n-layers 3 --fanout 50 --n-epoch 100 --early_stopping 10 --validation True --lr 0.05 --batch-size 256 --model-id 'GAT' --model 'GAT' --optimizer 'Adam' --device-id '5'"
elif [ $1 = "GATv2" ]
then echo "python train.py --model-id 'GATv2' --model 'GATv2' --dataset './dataset/pyg_data/icdm2022_session1.pt' --h-dim 8 --n-bases 8 --n-layers 3 --fanout 50 --n-epoch 100 --early_stopping 10 --validation True --lr 0.05 --batch-size 256 --optimizer 'Adam' --device-id '5'"
elif [ $1 = "Transformer" ]
then echo "python train.py --model-id 'Transformer' --model 'Transformer' --dataset './dataset/pyg_data/icdm2022_session1.pt' --h-dim 256 --n-bases 8 --n-layers 3 --fanout 50 --n-epoch 100 --early_stopping 10 --validation True --lr 0.01 --batch-size 256 --optimizer 'Adam' --device-id '6'"
elif [ $1 = "TAG" ]
then echo "python train.py --model-id 'TAG' --model 'TAG' --dataset './dataset/pyg_data/icdm2022_session1.pt' --h-dim 256 --n-bases 8 --n-layers 3 --fanout 50 --n-epoch 100 --early_stopping 10 --validation True --lr 0.01 --batch-size 256 --optimizer 'Adam' --device-id '6'"
elif [ $1 = "ARMA" ]
then echo "python train.py --model-id 'ARMA' --model 'ARMA' --dataset './dataset/pyg_data/icdm2022_session1.pt' --h-dim 256 --n-bases 8 --n-layers 2 --fanout 50 --n-epoch 100 --early_stopping 10 --validation True --lr 0.01 --batch-size 256 --optimizer 'Adam' --device-id '6'"
elif [ $1 = "SG" ]
then echo "python train.py --model-id 'SG' --model 'SG' --dataset './dataset/pyg_data/icdm2022_session1.pt' --h-dim 256 --n-bases 8 --n-layers 3 --fanout 50 --n-epoch 100 --early_stopping 10 --validation True --lr 0.01 --batch-size 256 --optimizer 'Adam' --device-id '6'"
elif [ $1 = "MF" ]
then echo "python train.py --model-id 'MF' --model 'MF' --dataset './dataset/pyg_data/icdm2022_session1.pt' --h-dim 256 --n-bases 8 --n-layers 3 --fanout 50 --n-epoch 100 --early_stopping 10 --validation True --lr 0.01 --batch-size 256 --optimizer 'Adam' --device-id '6'"
elif [ $1 = "EG" ]
then echo "python train.py --model-id 'EG' --model 'EG' --dataset './dataset/pyg_data/icdm2022_session1.pt' --h-dim 256 --n-bases 8 --n-layers 3 --fanout 50 --n-epoch 100 --early_stopping 10 --validation True --lr 0.01 --batch-size 256 --optimizer 'Adam' --device-id '6'"
elif [ $1 = "superGAT" ]
then echo "python train.py --model-id 'superGAT' --model 'superGAT' --dataset './dataset/pyg_data/icdm2022_session1.pt' --h-dim 8 --n-bases 8 --n-layers 2 --fanout 50 --n-epoch 100 --early_stopping 10 --validation True --lr 0.05 --batch-size 256 --optimizer 'Adam' --device-id '5'"
fi