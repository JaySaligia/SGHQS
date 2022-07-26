# Parameters

#######
# Epoch 7  (rgcn_flag_2)
# Trick Flag
# random seed 4399
# score 0.933659
#######

#######
# Epoch 8  (rgcn_flag_4)
# Trick Flag
# random seed 4399
# score 0.9341
#######

#######
# Epoch 8
# Trick Flag + PairNorm
# random seed 4399
# score 0.934026
#######

model_id="rgcn_sample"
gpu_id=1
icdm_sesison1_dir="/data/pengmiao/ICDM_dataset/icdm2022_session1_train/"
ouput_result_dir="/data/pengmiao/workplace/pycharm/icdm_graph_competition/pyg_example/"
pyg_data_session1="/data/pengmiao/ICDM_dataset/pyg_data/icdm2022_session1"
test_ids_session1="/data/pengmiao/ICDM_dataset/icdm2022_session1_test_ids.txt"

#ouput_result_dir="/data/pengmiao/workplace/pycharm/icdm_graph_competition/pyg_example/"
#pyg_data_session1="/data/liuben/icdm_dataset/icdm2022_session1"
#test_ids_session1="/data/pengmiao/ICDM_dataset/icdm2022_session1_test_ids.txt"

# Model hyperparameters
h_dim=256
n_bases=8
num_layers=3
fanout=150  # 150
n_epoch=10
early_stopping=6
lr=0.001
batch_size=200
sample=True
flag_use=False

# sesison1 data generator
#python format_pyg.py --graph=$icdm_sesison1_dir"icdm2022_session1_edges.csv" \
#        --node=$icdm_sesison1_dir"icdm2022_session1_nodes.csv" \
#        --label=$icdm_sesison1_dir"icdm2022_session1_train_labels.csv" \
#        --storefile=$pyg_data_session1

## Test: check if everything is correct
#python rgcn_sage_icdm.py --dataset  $pyg_data_session1".pt" \
#              --h-dim 10 \
#              --n-bases 8 \
#              --n-layers 2 \
#              --fanout 50 \
#              --n-epoch 3 \
#              --early_stopping 1 \
#              --validation True \
#              --lr 0.01 \
#              --batch-size 1000 \
#              --model-id "test" \
#              --device-id $gpu_id

# Training: session 1 (save model at best_models/$model_id.pth)
python rgcn_flag.py --dataset $pyg_data_session1".pt" \
               --h-dim $h_dim \
               --n-bases $n_bases \
               --n-layers $num_layers \
               --fanout $fanout \
               --n-epoch $n_epoch \
               --early_stopping $early_stopping \
               --validation True \
               --lr $lr \
               --batch-size $batch_size \
               --model-id $model_id \
               --device-id $gpu_id \
               --sample $sample \
               --flag_use $flag_use


# Inference: session1 1. loading model $model_id 2. reading test_ids 3. generator .json file
python rgcn_flag.py --dataset $pyg_data_session1".pt" \
        --test-file $test_ids_session1 \
        --json-file $ouput_result_dir"rgcn_flag_xwj_4399_epoch9.json" \
        --batch-size $batch_size \
        --n-layers $num_layers \
        --fanout $fanout \
        --inference True \
        --model-id $model_id \
        --device-id $gpu_id \
        --sample $sample \
        --flag_use $flag_use
