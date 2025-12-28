# -*- coding: utf-8 -*-
"""
HCCF实验对比脚本
在Grocery_and_Gourmet_Food数据集上对比HCCF与BPRMF、NeuMF、LightGCN
"""
import os
import sys
import subprocess
import argparse

# 设置路径
RECHORUS_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(RECHORUS_ROOT, 'src')

# 实验配置
MODELS = ['BPRMF', 'NeuMF', 'LightGCN', 'HCCF']
DATASETS = ['Grocery_and_Gourmet_Food']  # MovieLens_1M需要先处理

# 通用参数
COMMON_ARGS = {
    'emb_size': 64,
    'lr': 1e-3,
    'l2': 1e-6,
    'batch_size': 256,
    'epoch': 100,
    'early_stop': 10,
    'num_neg': 1,
    'test_all': 0,
    'path': '../data/',  # 数据路径
}

# 模型特定参数
MODEL_ARGS = {
    'BPRMF': {},
    'NeuMF': {},
    'LightGCN': {
        'n_layers': 3,
    },
    'HCCF': {
        'n_layers': 2,
        'temp': 0.2,
        'ssl_reg': 0.1,
        'keep_rate': 0.5,
    },
}

def run_experiment(model, dataset, gpu=0):
    """运行单个实验"""
    cmd = [
        sys.executable, 'main.py',
        '--model_name', model,
        '--dataset', dataset,
        '--gpu', str(gpu),
    ]
    
    # 添加通用参数
    for key, value in COMMON_ARGS.items():
        cmd.extend([f'--{key}', str(value)])
    
    # 添加模型特定参数
    if model in MODEL_ARGS:
        for key, value in MODEL_ARGS[model].items():
            cmd.extend([f'--{key}', str(value)])
    
    print(f"\n{'='*60}")
    print(f"Running: {model} on {dataset}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    # 运行命令
    result = subprocess.run(cmd, cwd=SRC_DIR, capture_output=False, text=True)
    return result.returncode

def main():
    parser = argparse.ArgumentParser(description='HCCF Experiment Script')
    parser.add_argument('--models', nargs='+', default=MODELS, help='Models to run')
    parser.add_argument('--datasets', nargs='+', default=DATASETS, help='Datasets to use')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    args = parser.parse_args()
    
    results = []
    
    for dataset in args.datasets:
        for model in args.models:
            print(f"\n>>> Running {model} on {dataset}...")
            returncode = run_experiment(model, dataset, args.gpu)
            results.append({
                'model': model,
                'dataset': dataset,
                'status': 'success' if returncode == 0 else 'failed'
            })
    
    # 打印结果摘要
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    for r in results:
        print(f"{r['model']:15} | {r['dataset']:30} | {r['status']}")
    print("="*60)
    print("\n查看 src/log/ 目录获取详细结果")

if __name__ == '__main__':
    main()
