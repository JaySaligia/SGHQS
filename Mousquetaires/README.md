 <p align="center">
    <img src="https://www.hualigs.cn/image/6300e0e23c6f1.jpg" height="150">
</p>

<h1 align="center">
    三个火枪手
</h1>

## 方法介绍

我们团队所提出的方法主要分为三个部分：
#### 1. 基于关系类型的子图采样器
由于电商图是一个极度类别平衡的大规模异质图，考虑到商品图的schema，商品之间的拓扑关联较为稀疏且为了适应测试阶段的inductive设定，因此采用基于one-hop关系类型的采样方式。 

具体对于一个target节点，为了保证节点的邻域信息均衡，我们对其每种关系类型的邻域有放回的采样相同数目的节点，然后迭代K次以获得一个子图。
#### 2. R-GCN编码器
采用R-GCN编码器作为节点的聚合器，具体的在聚合邻居节点的信息时按照边的类型进行分类，根据边类型的不同进行相应的转换，其中每个节点的信息更新共享参数，并行计算。
#### 3. 基于梯度扰动的对抗训练策略
为了更好的适应存在噪声数据的真实场景，避免由于小的噪声导致检测失败，考虑通过基于特征的数据增强方法来提高模型的鲁棒性。受到FLAG启发，对抗扰动被认为是一种数据依赖的正则化，有助于推广到分布外样本，同时考虑到标签节点样本的稀缺性，我们采用对抗扰动策略作为输入特征增强的方法。


### 仓库目录
```markdown
├── code
│   ├── flag.py        -- Free Large-scale Adversarial Augmentation on Graphs
│   ├── format_pyg.py  -- Pyg数据预处理代码
│   ├── main.py        -- 模型运行文件
│   ├── session1.sh    -- Session1运行命令及参数配置
│   ├── session2.sh    -- Session2运行命令及参数配置
├── data
│   ├── other          -- 中间件结果存放的目录
│   ├── session1       -- Session1数据集文件目录
│   ├── session2       -- Session2数据集文件目录
├── submit             -- 结果json文件保存目录
├── README.md
├── requirements.txt
```

## 环境依赖

### Environmental Settings
- Linux Ubuntu 18.04.6 LTS 
- NVIDIA TITAN RTX(24GB)
- CUDA 10.2
- CUDNN 8.2.2

### Python Packages
- Python 3.9.0
- Numpy 1.22.4
- Pytorch 1.11.0
  - torch-cluster 1.6.0 
  - torch-geometric 2.0.4
  - torch-scatter 2.0.9 
  - torch-sparse 0.6.14
- Scikit-learn 1.1.1
- tqdm 4.64.0

## 运行方法

### 环境安装

```shell
conda create -n SGHQS python=3.9
conda activate SGHQS
pip install -r $PATH_TO_PROJECT/requirements.txt
```

---
> Tips: 因为无需提交session1的训练集和测试集，考虑到数据处理阶段的超参可能与官方不一致，因此生成结果可能有细微差距。
运行时的超参best_epoch为我们在自己生成的数据上的最优epoch，考虑到以上问题，在官方生成的数据上可能不是最优。

### Session1

运行session1.sh文件：

#### 1. 读取数据集，处理成pyg格式文件并保存

```shell
cd $PATH_TO_PROJECT/code/

python format_pyg.py --graph=$icdm_sesison1_dir"icdm2022_session1_edges.csv" \
        --node=$icdm_sesison1_dir"icdm2022_session1_nodes.csv" \
        --label=$icdm_sesison1_dir"icdm2022_session1_train_labels.csv" \
        --storefile=$pyg_data_session1
```
Output:

PyG Graph放置在`$PATH_TO_PROJECT/data/session1/`目录下

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
               --device-id $gpu_id \
               --session $session_type \
               --best_epcoh $best_epoch
```
Output:

训练好的模型权重文件放置在`$PATH_TO_PROJECT/data/other/`目录下

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

最终的测试集推断结果保存在`$PATH_TO_PROJECT/submit/`目录下

- submit_Ymd_HMS.json: 模型在测试集上的推断结果

---
### Session2

运行session2.sh文件：

#### 1. 读取数据集，处理成pyg格式文件并保存

```shell
cd $PATH_TO_PROJECT/code/

# sesison2 data generator
python format_pyg.py --graph=$icdm_sesison2_dir"icdm2022_session2_edges.csv" \
        --node=$icdm_sesison2_dir"icdm2022_session2_nodes.csv" \
        --label=$icdm_sesison2_dir"icdm2022_session2_train_labels.csv" \
        --storefile=$pyg_data_session2
```
Output:

PyG Graph放置在`$PATH_TO_PROJECT/data/session2/`目录下

- icdm2022_session2.pt: 预处理后session2的Pyg格式的异质图文件
- *.nodes.pyg: 处理过程中的临时存储文件

#### 2. 训练模型

```shell
# Training: session 2 (save model at data/other/$model_id.pth)
python main.py --dataset $pyg_data_session2".pt" \
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
               --best_epcoh $best_epoch
```
Output:

训练好的模型权重文件放置在`$PATH_TO_PROJECT/data/other/`目录下

- $Model_id.pth: 模型权重文件

#### 3. 推断结果

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

最终的测试集推断结果保存在`$PATH_TO_PROJECT/submit/`目录下

- submit_Ymd_HMS.json: 模型在测试集上的推断结果

