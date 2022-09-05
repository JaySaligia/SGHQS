# Parameters

model_id="model_session2"
gpu_id=5
session_type=2  # 1 or 2
icdm_sesison2_dir="../data/session2/"
pyg_data_session1="../data/session1/icdm2022_session1"
pyg_data_session2="../data/session2/icdm2022_session2"
test_ids_session2="../data/session2/icdm2022_session2_test_ids.txt"


# Model hyperparameters
h_dim=256
n_bases=8
num_layers=3
fanout=150  # 150
n_epoch=100
early_stopping=12
lr=0.001
batch_size=200
best_epoch=7

# sesison2 data generator
python format_pyg.py --graph=$icdm_sesison2_dir"icdm2022_session2_edges.csv" \
        --node=$icdm_sesison2_dir"icdm2022_session2_nodes.csv" \
        --storefile=$pyg_data_session2

# Training: session 2 (save model at data/other/$model_id.pth)
python main.py --dataset $pyg_data_session1".pt" \
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
               --session $session_type \
               --best_epoch $best_epoch


# Inference: session2 1. loading model $model_id 2. reading test_ids 3. generator .json file
python main.py --dataset $pyg_data_session2".pt" \
        --test-file $test_ids_session2 \
        --batch-size $batch_size \
        --n-layers $num_layers \
        --fanout $fanout \
        --inference True \
        --model-id $model_id \
        --device-id $gpu_id

