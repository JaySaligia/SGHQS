if [ $1 = "RGCN" ]
then echo "python train.py --dataset './dataset/pyg_data/icdm2022_session1.pt' --h-dim 10 --n-bases 8 --n-layers 2 --fanout 50 --n-epoch 3 --early_stopping 1 --validation True --lr 0.01 --batch-size 1000 --model-id 'RGCN' --model 'RGCN' --optimizer 'Adam' --device-id 6"
elif [ $1 = "HGT" ]
then echo "python train.py --dataset './dataset/pyg_data/icdm2022_session1.pt' --h-dim 512 --n-bases 8 --n-layers 2 --fanout 150 --n-epoch 100 --early_stopping 6 --validation True --lr 0.001 --batch-size 512 --model-id 'hgt_4' --model 'HGT' --optimizer 'Adam' --device-id 6"
fi