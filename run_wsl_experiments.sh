#!/bin/bash
# HCCF实验 - WSL环境配置和运行脚本
# 使用方法: 在WSL中运行 bash run_wsl_experiments.sh

echo "=============================================="
echo "HCCF 机器学习大作业 - WSL+GPU 实验"
echo "=============================================="

# 设置路径 - 根据你的WSL挂载点修改
WINDOWS_PATH="/mnt/c/Users/王唱晓/Downloads/ReChorus-master/ReChorus-master"
cd "$WINDOWS_PATH"

# 1. 创建虚拟环境
echo "[1/6] 创建Python虚拟环境..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

# 2. 安装依赖
echo "[2/6] 安装依赖..."
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas scipy scikit-learn tqdm pyyaml

# 3. 检查GPU
echo "[3/6] 检查GPU..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# 4. 进入src目录
cd src

# 5. 运行实验
echo "[4/6] 开始运行实验..."

# Grocery数据集实验
DATASET="Grocery_and_Gourmet_Food"
DATA_PATH="../data/"

echo ">>> 运行 BPRMF on $DATASET..."
python main.py --model_name BPRMF --emb_size 64 --lr 1e-3 --l2 1e-6 --dataset $DATASET --path $DATA_PATH --epoch 100 --gpu 0

echo ">>> 运行 NeuMF on $DATASET..."
python main.py --model_name NeuMF --emb_size 64 --lr 1e-3 --l2 1e-6 --dataset $DATASET --path $DATA_PATH --epoch 100 --gpu 0

echo ">>> 运行 LightGCN on $DATASET..."
python main.py --model_name LightGCN --emb_size 64 --n_layers 3 --lr 1e-3 --l2 1e-6 --dataset $DATASET --path $DATA_PATH --epoch 100 --gpu 0

echo ">>> 运行 HCCF on $DATASET..."
python main.py --model_name HCCF --emb_size 64 --n_layers 2 --temp 0.2 --ssl_reg 0.1 --lr 1e-3 --l2 1e-6 --dataset $DATASET --path $DATA_PATH --epoch 100 --gpu 0

echo "[5/6] Grocery实验完成！"

# 6. 汇总结果
echo "[6/6] 实验结果汇总:"
echo "请查看 log/ 目录下的日志文件获取详细结果"
ls -la ../log/

echo "=============================================="
echo "实验完成！"
echo "=============================================="
