 <p align="center">
    <img src="https://www.hualigs.cn/image/6300e0e23c6f1.jpg" height="150">
</p>

<h1 align="center">
    三个火枪手
</h1>

## 方法介绍


## 环境依赖

### Environmental Settings
- Linux Ubuntu 18.04.6 LTS 
- NVIDIA TITAN RTX(24GB)
- CUDA 10.2
- CUDNN 8.2.2

### Python Packages
- Python 3.9.0
- Numpy 1.22.4
- Pandas 1.4.3
- Pytorch 1.11.0
  - torch-cluster 1.6.0 
  - torch-geometric 2.0.4
  - torch-scatter 2.0.9 
  - torch-sparse 0.6.14
- NetworkX 2.8.5
- Scikit-learn 1.1.1

## 运行方法

### 环境安装

```shell
conda create -n SGHQS python=3.9
conda activate SGHQS
pip install -r $PATH_TO_PROJRCT/requirements.txt
```

---
### Session1

运行session1.sh文件：

#### 1. 读取数据集，处理成pyg格式文件并保存

```shell
cd $PATH_TO_PROJRCT/code/

python format_pyg.py --graph=$icdm_sesison1_dir"icdm2022_session1_edges.csv" \
        --node=$icdm_sesison1_dir"icdm2022_session1_nodes.csv" \
        --label=$icdm_sesison1_dir"icdm2022_session1_train_labels.csv" \
        --storefile=$pyg_data_session1
```
Output:

PyG Graph放置在`$PATH_TO_PROJRCT/data/session1/`目录下

- icdm2022_session1.pt: 预处理后session1的Pyg格式的异质图文件
- *.nodes.pyg: 处理过程中的临时存储文件

#### 2. 训练模型

```shell
# Training: session 1 (save model at data/other/$model_id.pth)
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
               --device-id $gpu_id
```
Output:

训练好的模型权重文件放置在`$PATH_TO_PROJRCT/data/other/`目录下

- $Model_id.pth: 模型权重文件

#### 3. 推断结果

```shell
# Inference: session1 1. loading model $model_id 2. reading test_ids 3. generator .json file
python main.py --dataset $pyg_data_session1".pt" \
        --test-file $test_ids_session1 \
        --batch-size $batch_size \
        --n-layers $num_layers \
        --fanout $fanout \
        --inference True \
        --model-id $model_id \
        --device-id $gpu_id
```
Output:

最终的测试集推断结果保存在`$PATH_TO_PROJRCT/submit/`目录下

- submit_Ymd_HMS.json: 模型在测试集上的推断结果

---
### Session2

运行session2.sh文件：

#### 1. 读取数据集，处理成pyg格式文件并保存

```shell
cd $PATH_TO_PROJRCT/code/

# sesison2 data generator
python format_pyg.py --graph=$icdm_sesison2_dir"icdm2022_session2_edges.csv" \
        --node=$icdm_sesison2_dir"icdm2022_session2_nodes.csv" \
        --label=$icdm_sesison2_dir"icdm2022_session2_train_labels.csv" \
        --storefile=$pyg_data_session2
```
Output:

PyG Graph放置在`$PATH_TO_PROJRCT/data/session2/`目录下

- icdm2022_session2.pt: 预处理后session2的Pyg格式的异质图文件
- *.nodes.pyg: 处理过程中的临时存储文件

#### 2. 推断结果

```shell
# Inference: session2 1. loading model $model_id 2. reading test_ids 3. generator .json file
python main.py --dataset $pyg_data_session2".pt" \
        --test-file $test_ids_session2 \
        --batch-size $batch_size \
        --n-layers $num_layers \
        --fanout $fanout \
        --inference True \
        --model-id $model_id \
        --device-id $gpu_id
```
Output:

最终的测试集推断结果保存在`$PATH_TO_PROJRCT/submit/`目录下

- submit_Ymd_HMS.json: 模型在测试集上的推断结果

