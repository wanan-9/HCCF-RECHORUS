# HCCF算法复现 - 机器学习大作业

## 项目说明

本项目在ReChorus框架中复现了HCCF (Hypergraph Contrastive Collaborative Filtering) 算法。

**论文**: Hypergraph Contrastive Collaborative Filtering (SIGIR 2022)  
**arXiv**: https://arxiv.org/abs/2204.12200

## 目录结构

```
ReChorus-master/
├── src/
│   ├── models/
│   │   └── general/
│   │       └── HCCF.py          # HCCF模型实现
│   └── main.py                  # 主程序入口
├── data/
│   ├── Grocery_and_Gourmet_Food/  # Amazon数据集
│   └── MovieLens_1M/              # MovieLens数据集(需处理)
└── run_experiments.py           # 实验运行脚本
```

## 环境配置

```bash
# 1. 创建虚拟环境
python -m venv hccf_env

# 2. 激活环境
hccf_env\Scripts\activate  # Windows
# source hccf_env/bin/activate  # Linux/Mac

# 3. 安装依赖
pip install torch numpy pandas scipy scikit-learn tqdm pyyaml
```

## 运行实验

### 方式1: 单独运行模型

```bash
cd src

# 运行HCCF
python main.py --model_name HCCF --emb_size 64 --n_layers 2 --temp 0.2 --ssl_reg 0.1 --lr 1e-3 --l2 1e-6 --dataset Grocery_and_Gourmet_Food --path ../data/

# 运行对比算法
python main.py --model_name BPRMF --emb_size 64 --lr 1e-3 --l2 1e-6 --dataset Grocery_and_Gourmet_Food --path ../data/
python main.py --model_name NeuMF --emb_size 64 --lr 1e-3 --l2 1e-6 --dataset Grocery_and_Gourmet_Food --path ../data/
python main.py --model_name LightGCN --emb_size 64 --n_layers 3 --lr 1e-3 --l2 1e-6 --dataset Grocery_and_Gourmet_Food --path ../data/
```

### 方式2: 批量运行实验

```bash
cd ReChorus-master
python run_experiments.py
```

## HCCF超参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| emb_size | 64 | 嵌入向量维度 |
| n_layers | 2 | GNN层数 |
| temp | 0.2 | 对比学习温度系数 |
| ssl_reg | 0.1 | 对比学习损失权重 |
| keep_rate | 0.5 | 边dropout保留率 |
| leaky | 0.5 | LeakyReLU斜率 |

## 评价指标

- **HR@K**: Hit Rate，正例是否出现在Top-K列表中
- **NDCG@K**: Normalized Discounted Cumulative Gain，排序质量

## 实验结果

结果保存在 `src/log/` 目录下。

## 作者

机器学习大作业 - HCCF复现项目
王唱晓、王鹏华
