# -*- coding: utf-8 -*-
"""
HCCF Ablation Study Script
用于分析 HCCF 各模块对整体性能的贡献
"""

import os
import sys
import subprocess

RECHORUS_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(RECHORUS_ROOT, 'src')

DATASET = 'Grocery_and_Gourmet_Food'
GPU = 0

COMMON_ARGS = {
    'emb_size': 64,
    'lr': 1e-3,
    'l2': 1e-6,
    'batch_size': 256,
    'epoch': 100,
    'early_stop': 10,
    'num_neg': 1,
    'test_all': 0,
    'path': '../data/',
}

# ===============================
# 消融配置
# ===============================
ABLATION_CONFIGS = {
    'HCCF_full': {
        'n_layers': 2,
        'temp': 0.2,
        'ssl_reg': 0.1,
        'keep_rate': 0.5,
    },
    'HCCF_wo_SSL': {
        'n_layers': 2,
        'temp': 0.2,
        'ssl_reg': 0.0,
        'keep_rate': 0.5,
    },
    'HCCF_wo_Dropout': {
        'n_layers': 2,
        'temp': 0.2,
        'ssl_reg': 0.1,
        'keep_rate': 1.0,
    },
    'HCCF_1layer': {
        'n_layers': 1,
        'temp': 0.2,
        'ssl_reg': 0.1,
        'keep_rate': 0.5,
    },
    'HCCF_wo_SSL_Dropout': {
        'n_layers': 2,
        'temp': 0.2,
        'ssl_reg': 0.0,
        'keep_rate': 1.0,
    }
}

def run_ablation():
    for name, args in ABLATION_CONFIGS.items():
        cmd = [
            sys.executable, 'main.py',
            '--model_name', 'HCCF',
            '--dataset', DATASET,
            '--gpu', str(GPU),
            '--exp_name', name,  # 非常重要：区分日志
        ]

        for k, v in COMMON_ARGS.items():
            cmd.extend([f'--{k}', str(v)])

        for k, v in args.items():
            cmd.extend([f'--{k}', str(v)])

        print(f'\n==== Running Ablation: {name} ====')
        subprocess.run(cmd, cwd=SRC_DIR)

if __name__ == '__main__':
    run_ablation()
