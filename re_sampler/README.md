 <p align="center">
    <img src="https://www.hualigs.cn/image/6300e0e23c6f1.jpg" height="150">
</p>

<h1 align="center">
    Re-Sampling解决方案
</h1>

<p align="center">
    针对赛题数据存在的挑战，尝试从采样方案优化的方式去解决，最后的结果一般，仅当做记录，希望对以后解决这类问题提供一些帮助。
</p>

## Challenge

通过对初赛数据进行分析可以发现，该赛事存在以下两个挑战：

- 异常检测任务常见的挑战，对于target nodes，类别及其不平衡，多数类（正样本）是少数类（负样本）的**9**倍多，会导致模型在训练的时候更倾向于将少数类分为多数类
- 第二个挑战在于大多数图神经网络都是适用于homophily graph, 也就是说具有同样类型的节点在空间结构上更相近，而在异常检测的场景中，异常节点通过伪装的方式使得其具有较为”干净“的关联关系，如何去选择好的neighbor是一大挑战。

因为考虑到复赛给的节点是inductive setup,所以我们并不是学习nn.embedding，而是希望学习一个好的graph network分类器，所以在本这里实现的过程中，我们在初始读入nn.featrue后，会将required_grad=False.

```python
import torch.nn as nn

features = nn.Embedding(feat_data.shape[0], feat_data.shape[1])
features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
```

## Resources

- 在github上找到了收集好的ICML、IJCAI、WWW等顶会上关于异常检测的论文：https://github.com/safe-graph/graph-fraud-detection-papers

- 该仓库提供了Graph上异常检测开箱即用的工具包：

    - [PyGOD: A Python Library for Graph Outlier Detection (Anomaly Detection)](https://github.com/pygod-team/pygod)

    - [DGFraud: A Deep Graph-based Toolbox for Fraud Detection](https://github.com/safe-graph/DGFraud)

    - [UGFraud: An Unsupervised Graph-based Toolbox for Fraud Detection](https://github.com/safe-graph/UGFraud)

    - [GNN-based Fake News Detection](https://github.com/safe-graph/GNN-FakeNews)

    - [Realtime Fraud Detection with GNN on DGL](https://github.com/awslabs/realtime-fraud-detection-with-gnn-on-dgl)

## Implementation

考虑到样本类型的极度不平衡，我们希望模型在采样训练的时候，能够对类别较少的target nodes采集更多的neighbors，而对类别较多的target nodes进行under-sampling。换句话说，就是对负样本的语义、结构更多的关注。具体而言，首先我们并非对所有的训练node进行采样训练，而是只用关注对label影响比较大的训练node，受Personalized PageRank启发，选择训练节点的概率取决于它的度以及其所属类别的频率。

$$P(v) \propto \frac{\big\|A(:,v)\big\|^2}{Sum(C_{v \in c})}$$

在选择一定数量的训练样本后（这里我定的是两倍的minor class数量），将这些节点作为真实的训练的节点，然后对每个节点我们希望在采样的时候尽可能的考虑到节点的label，我们首先计算target node的所有neighbor的label score，然后将所有neighbor的label score和target node的label score进行比较并排序，如果target node是major class，则对其进行under sample，采样top-k个neighbor，如果target node是minor class, 则对其进行一定的over sample根据相应的neighbor的label score。最终模型有两个loss，一个是label-aware loss用来学习label transform weight，来选择和target node的label score相近的neighbor，另外一个loss是GNN loss，也就是通过GNN学习到的最终的node score与真实label的loss。

## Conclusions

开始时，选择一定的node加上induced sample发现效果并不好，后来想一想也理所当然，因为induce sample的局限性，后面采用label-aware sample后，也就是对major class node和minor class node进行re-sampling，虽然相比之前的induced-sample有一定的提升，但是依然不理想，可能存在的原因是targe node的one-hop节点类型仅有两个，当然还有可能是这个方法本来就是歪的，所以不管了，就当记录，希望对复赛以及可能以后面对这类问题有一个启发。