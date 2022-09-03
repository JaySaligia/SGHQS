# Parameters


model_id="rgcn_flag_10"  # rgcn_flag_9
gpu_id=9

icdm_sesison2_dir="../data/session2/"
pyg_data_session2="../data/session2/icdm2022_session2"
test_ids_session2="../data/session2/icdm2022_session2_test_ids.txt"


# Model hyperparameters
#h_dim=256
#n_bases=8
num_layers=3
fanout=150  # 150
#n_epoch=100
#early_stopping=6
#lr=0.001
batch_size=200

# sesison2 data generator
python format_pyg.py --graph=$icdm_sesison2_dir"icdm2022_session2_edges.csv" \
        --node=$icdm_sesison2_dir"icdm2022_session2_nodes.csv" \
        --label=$icdm_sesison2_dir"icdm2022_session2_train_labels.csv" \
        --storefile=$pyg_data_session2


# Inference: session2 1. loading model $model_id 2. reading test_ids 3. generator .json file
python main.py --dataset $pyg_data_session2".pt" \
        --test-file $test_ids_session2 \
        --batch-size $batch_size \
        --n-layers $num_layers \
        --fanout $fanout \
        --inference True \
        --model-id $model_id \
        --device-id $gpu_id

