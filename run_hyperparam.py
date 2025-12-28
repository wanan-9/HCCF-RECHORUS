# -*- coding: utf-8 -*-
"""
HCCF Hyper-parameter Study Script
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

BASE_HCCF_ARGS = {
    'n_layers': 2,
    'temp': 0.2,
    'ssl_reg': 0.1,
    'keep_rate': 0.5,
}

# ===============================
# 超参搜索空间
# ===============================
HYPER_PARAMS = {
    'ssl_reg': [0.0, 0.01, 0.05, 0.1, 0.2],
    'temp': [0.1, 0.2, 0.5, 1.0],
    'keep_rate': [0.3, 0.5, 0.7, 1.0],
    'n_layers': [1, 2, 3],
}

def run_hyper():
    for param, values in HYPER_PARAMS.items():
        for v in values:
            args = BASE_HCCF_ARGS.copy()
            args[param] = v

            exp_name = f'HCCF_{param}_{v}'
            cmd = [
                sys.executable, 'main.py',
                '--model_name', 'HCCF',
                '--dataset', DATASET,
                '--gpu', str(GPU),
                '--exp_name', exp_name,
            ]

            for k, val in COMMON_ARGS.items():
                cmd.extend([f'--{k}', str(val)])

            for k, val in args.items():
                cmd.extend([f'--{k}', str(val)])

            print(f'\n==== Hyper-param: {param} = {v} ====')
            subprocess.run(cmd, cwd=SRC_DIR)

if __name__ == '__main__':
    run_hyper()
