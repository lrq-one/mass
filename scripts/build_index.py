import torch
import pandas as pd
import pickle
import os
import numpy as np
from tqdm import tqdm
from pathlib import Path

def build_reference_index(tsv_path, output_path):
    """
    读取 MassSpecGym.tsv 并构建 InChIKey -> {mz, intensity, smiles} 的离线索引。
    """
    print(f"Loading metadata from {tsv_path}...")
    df = pd.read_csv(tsv_path, sep="\t")
    
    # 过滤掉缺失关键信息的行
    # 优先使用训练集作为参考库
    if 'fold' in df.columns:
        # 这里可以根据需求选择参考库的范围，通常使用 train 集合
        ref_df = df[df['fold'] != 'test'] 
    else:
        ref_df = df

    index = {}
    print("Building index...")
    for _, row in tqdm(ref_df.iterrows(), total=len(ref_df)):
        # 使用 InChIKey 作为 ID，如果没有则用 SMILES
        # MassSpecGym 通常有名为 'inchikey' 的列，如果没有则实时计算
        ikey = row.get('inchikey')
        if pd.isna(ikey) or not ikey:
            # 简单回退处理
            ikey = row['smiles']
            
        # 解析谱图数据
        mzs = np.array([float(m) for m in str(row['mzs']).split(',')])
        ints = np.array([float(i) for i in str(row['intensities']).split(',')])
        
        # 强度归一化 (0-1)
        if len(ints) > 0:
            ints = ints / (np.max(ints) + 1e-8)
            
        index[ikey] = {
            'mz': mzs.astype(np.float32),
            'intensity': ints.astype(np.float32),
            'smiles': row['smiles'],
            'precursor_mz': float(row['precursor_mz'])
        }
        
    print(f"Saving index with {len(index)} entries to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(index, f)
    print("Done.")

if __name__ == "__main__":
    # 自动探测 MassSpecGym.tsv 的位置
    possible_paths = [
        "rassp-public-main/data/MassSpecGym.tsv",
        "data/MassSpecGym.tsv",
        "MassSpecGym.tsv",
        "../rassp-public-main/data/MassSpecGym.tsv"
    ]
    
    tsv_file = None
    for p in possible_paths:
        if os.path.exists(p):
            tsv_file = p
            break
            
    if tsv_file:
        print(f"Found dataset at: {tsv_file}")
        output_file = "reference_spectra_index.pkl"
        build_reference_index(tsv_file, output_file)
    else:
        print("Error: Could not find MassSpecGym.tsv in common locations.")
        print(f"Searched: {possible_paths}")
