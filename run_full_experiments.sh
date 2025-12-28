#!/bin/bash
# HCCF完整实验 - WSL+GPU
# 包含: 环境配置、MovieLens处理、全部实验运行

echo "=============================================="
echo "HCCF 机器学习大作业 - 完整实验流程"
echo "=============================================="

# Windows路径转WSL路径
WINDOWS_USER="王唱晓"
BASE_PATH="/mnt/c/Users/${WINDOWS_USER}/Downloads/ReChorus-master/ReChorus-master"

cd "$BASE_PATH" || { echo "路径不存在: $BASE_PATH"; exit 1; }

# ========== 1. 环境配置 ==========
echo ""
echo "[1/5] 配置Python环境..."

# 创建虚拟环境
if [ ! -d "venv_gpu" ]; then
    python3 -m venv venv_gpu
    echo "虚拟环境创建成功"
fi

source venv_gpu/bin/activate

# 安装依赖
echo "安装PyTorch (CUDA 11.8)..."
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas scipy scikit-learn tqdm pyyaml

# 检查GPU
echo ""
echo "GPU检测结果:"
python3 -c "
import torch
print(f'  CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU设备: {torch.cuda.get_device_name(0)}')
    print(f'  GPU数量: {torch.cuda.device_count()}')
"

# ========== 2. 处理MovieLens数据 ==========
echo ""
echo "[2/5] 处理MovieLens-1M数据集..."
cd data
if [ ! -f "MovieLens_1M/train.csv" ]; then
    python3 process_movielens.py
else
    echo "MovieLens数据已处理，跳过"
fi
cd ..

# ========== 3. 运行Grocery实验 ==========
echo ""
echo "[3/5] 在Grocery数据集上运行实验..."
cd src

DATASET1="Grocery_and_Gourmet_Food"
COMMON_ARGS="--emb_size 64 --lr 1e-3 --l2 1e-6 --epoch 100 --gpu 0 --path ../data/"

echo ""
echo ">>> BPRMF on $DATASET1"
python main.py --model_name BPRMF --dataset $DATASET1 $COMMON_ARGS

echo ""
echo ">>> NeuMF on $DATASET1"
python main.py --model_name NeuMF --dataset $DATASET1 $COMMON_ARGS

echo ""
echo ">>> LightGCN on $DATASET1"
python main.py --model_name LightGCN --n_layers 3 --dataset $DATASET1 $COMMON_ARGS

echo ""
echo ">>> HCCF on $DATASET1"
python main.py --model_name HCCF --n_layers 2 --temp 0.2 --ssl_reg 0.1 --dataset $DATASET1 $COMMON_ARGS

# ========== 4. 运行MovieLens实验 ==========
echo ""
echo "[4/5] 在MovieLens-1M数据集上运行实验..."

DATASET2="MovieLens_1M"

echo ""
echo ">>> BPRMF on $DATASET2"
python main.py --model_name BPRMF --dataset $DATASET2 $COMMON_ARGS

echo ""
echo ">>> NeuMF on $DATASET2"
python main.py --model_name NeuMF --dataset $DATASET2 $COMMON_ARGS

echo ""
echo ">>> LightGCN on $DATASET2"
python main.py --model_name LightGCN --n_layers 3 --dataset $DATASET2 $COMMON_ARGS

echo ""
echo ">>> HCCF on $DATASET2"
python main.py --model_name HCCF --n_layers 2 --temp 0.2 --ssl_reg 0.1 --dataset $DATASET2 $COMMON_ARGS

# ========== 5. 汇总结果 ==========
echo ""
echo "[5/5] 实验结果汇总"
echo "=============================================="
echo "日志文件位置: $(pwd)/../log/"
echo ""
echo "结果摘要 (从日志中提取):"
echo ""

cd ../log
for f in *.log; do
    if [ -f "$f" ]; then
        echo "--- $f ---"
        tail -5 "$f" | grep -E "(HR|NDCG|Best)"
        echo ""
    fi
done

echo "=============================================="
echo "全部实验完成!"
echo "=============================================="
echo ""
echo "下一步: 根据log目录下的结果撰写实验报告"
