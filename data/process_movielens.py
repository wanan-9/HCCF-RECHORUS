# -*- coding: utf-8 -*-
"""
处理MovieLens-1M数据集为ReChorus格式
运行方法: python process_movielens.py
"""
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import urllib.request
import zipfile

# 配置
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
ML_DIR = os.path.join(DATA_DIR, 'MovieLens_1M')
RAW_DIR = os.path.join(ML_DIR, 'ml-1m')
OUTPUT_DIR = ML_DIR

RANDOM_SEED = 0
NEG_ITEMS = 99

def download_data():
    """下载MovieLens-1M数据集"""
    zip_path = os.path.join(ML_DIR, 'ml-1m.zip')
    
    if not os.path.exists(RAW_DIR):
        if not os.path.exists(zip_path):
            print("下载 MovieLens-1M 数据集...")
            url = "http://files.grouplens.org/datasets/movielens/ml-1m.zip"
            urllib.request.urlretrieve(url, zip_path)
            print("下载完成!")
        
        print("解压数据...")
        with zipfile.ZipFile(zip_path, 'r') as f:
            f.extractall(ML_DIR)
        print("解压完成!")
    else:
        print("数据已存在，跳过下载")

def load_ratings():
    """加载评分数据"""
    file_path = os.path.join(RAW_DIR, 'ratings.dat')
    
    interactions = []
    user_freq, item_freq = dict(), dict()
    
    print("读取评分数据...")
    with open(file_path, 'r', encoding='latin-1') as f:
        for line in tqdm(f):
            parts = line.strip().split('::')
            uid, iid, rating, time = parts[0], parts[1], float(parts[2]), int(parts[3])
            
            # 只保留正样本 (rating >= 4)
            if rating >= 4:
                interactions.append([int(uid), int(time), int(iid)])
                user_freq[uid] = user_freq.get(uid, 0) + 1
                item_freq[iid] = item_freq.get(iid, 0) + 1
    
    return interactions, user_freq, item_freq

def filter_5core(interactions, user_freq, item_freq):
    """5-core过滤: 用户和物品至少有5次交互"""
    print("5-core过滤...")
    
    while True:
        select_uid = {u for u, c in user_freq.items() if c >= 5}
        select_iid = {i for i, c in item_freq.items() if c >= 5}
        
        new_interactions = []
        user_freq, item_freq = dict(), dict()
        
        for uid, time, iid in interactions:
            if str(uid) in select_uid and str(iid) in select_iid:
                new_interactions.append([uid, time, iid])
                user_freq[str(uid)] = user_freq.get(str(uid), 0) + 1
                item_freq[str(iid)] = item_freq.get(str(iid), 0) + 1
        
        if len(new_interactions) == len(interactions):
            break
        interactions = new_interactions
    
    print(f"过滤后: {len(interactions)} 条交互, {len(user_freq)} 用户, {len(item_freq)} 物品")
    return interactions

def process_and_save(interactions):
    """处理数据并保存为ReChorus格式"""
    print("处理并保存数据...")
    
    # 创建DataFrame
    df = pd.DataFrame(interactions, columns=['user_id', 'time', 'item_id'])
    df = df.sort_values(by=['user_id', 'time']).reset_index(drop=True)
    
    # 重新编号用户和物品ID (从1开始)
    user_ids = sorted(df['user_id'].unique())
    item_ids = sorted(df['item_id'].unique())
    
    user2id = {u: i+1 for i, u in enumerate(user_ids)}
    item2id = {i: j+1 for j, i in enumerate(item_ids)}
    
    df['user_id'] = df['user_id'].map(user2id)
    df['item_id'] = df['item_id'].map(item2id)
    
    # 按时间分割: 80% train, 10% dev, 10% test
    # 使用leave-one-out: 每个用户最后一个交互作为test，倒数第二个作为dev
    train_data = []
    dev_data = []
    test_data = []
    
    print("分割训练/验证/测试集...")
    for user_id, group in tqdm(df.groupby('user_id')):
        group = group.sort_values('time')
        items = group['item_id'].tolist()
        times = group['time'].tolist()
        
        if len(items) >= 3:
            # 最后一个作为test
            test_data.append({
                'user_id': user_id,
                'item_id': items[-1],
                'time': times[-1]
            })
            # 倒数第二个作为dev
            dev_data.append({
                'user_id': user_id,
                'item_id': items[-2],
                'time': times[-2]
            })
            # 其余作为train
            for i in range(len(items) - 2):
                train_data.append({
                    'user_id': user_id,
                    'item_id': items[i],
                    'time': times[i]
                })
    
    train_df = pd.DataFrame(train_data)
    dev_df = pd.DataFrame(dev_data)
    test_df = pd.DataFrame(test_data)
    
    # 为dev和test生成负样本
    all_items = set(df['item_id'].unique())
    user_clicked = df.groupby('user_id')['item_id'].apply(set).to_dict()
    
    print("生成负样本...")
    np.random.seed(RANDOM_SEED)
    
    def generate_neg_items(row):
        user_id = row['user_id']
        clicked = user_clicked.get(user_id, set())
        neg_pool = list(all_items - clicked)
        if len(neg_pool) >= NEG_ITEMS:
            neg = np.random.choice(neg_pool, NEG_ITEMS, replace=False).tolist()
        else:
            neg = neg_pool + list(np.random.choice(list(all_items), NEG_ITEMS - len(neg_pool), replace=True))
        return neg
    
    dev_df['neg_items'] = dev_df.apply(generate_neg_items, axis=1)
    test_df['neg_items'] = test_df.apply(generate_neg_items, axis=1)
    
    # 保存
    print("保存数据...")
    train_df.to_csv(os.path.join(OUTPUT_DIR, 'train.csv'), sep='\t', index=False)
    dev_df.to_csv(os.path.join(OUTPUT_DIR, 'dev.csv'), sep='\t', index=False)
    test_df.to_csv(os.path.join(OUTPUT_DIR, 'test.csv'), sep='\t', index=False)
    
    print(f"完成! Train: {len(train_df)}, Dev: {len(dev_df)}, Test: {len(test_df)}")
    print(f"用户数: {df['user_id'].nunique()}, 物品数: {df['item_id'].nunique()}")

def main():
    print("=" * 50)
    print("处理 MovieLens-1M 数据集")
    print("=" * 50)
    
    # 1. 下载数据
    download_data()
    
    # 2. 加载评分
    interactions, user_freq, item_freq = load_ratings()
    
    # 3. 5-core过滤
    interactions = filter_5core(interactions, user_freq, item_freq)
    
    # 4. 处理并保存
    process_and_save(interactions)
    
    print("=" * 50)
    print("处理完成!")
    print("=" * 50)

if __name__ == '__main__':
    main()
