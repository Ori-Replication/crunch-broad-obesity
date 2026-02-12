"""
训练脚本：计算 Perturbmean baseline
- 计算所有扰动的平均 Delta（相对于 NC）
- 计算所有扰动的平均细胞比例
"""

import os
import gc
import numpy as np
import pandas as pd
import scanpy as sc
import joblib
from tqdm import tqdm


def train(
        data_directory_path: str,
        model_directory_path: str,
):
    print("=" * 80)
    print("Training Perturbmean Baseline")
    print("=" * 80)
    
    print("Loading training data (backed mode)...")
    file_path = os.path.join(data_directory_path, "obesity_challenge_1.h5ad")
    adata = sc.read_h5ad(file_path, backed="r")
    
    print(f"Data shape: {adata.shape}")
    
    # 读取细胞比例数据
    proportion_path = os.path.join(data_directory_path, "program_proportion.csv")
    if os.path.exists(proportion_path):
        df_prop_train = pd.read_csv(proportion_path)
        print(f"Loaded proportion data: {len(df_prop_train)} perturbations")
    else:
        print("Warning: program_proportion.csv not found. Will compute from data.")
        df_prop_train = None
    
    # 1. 提取控制组 (NC) 数据
    print("\nExtracting Control (NC) data...")
    nc_indices = np.where(adata.obs["gene"] == "NC")[0]
    print(f"NC cells: {len(nc_indices)}")
    
    # 读取 NC 细胞的表达数据
    X_nc_data = adata[nc_indices].X
    if hasattr(X_nc_data, "toarray"):
        X_nc = X_nc_data.toarray()
    else:
        X_nc = np.asarray(X_nc_data)
    
    # 计算 NC 的平均表达谱
    control_mean = np.ravel(np.asarray(np.mean(X_nc, axis=0)))
    print(f"Control mean shape: {control_mean.shape}")
    
    # 2. 计算所有扰动的平均 Delta
    print("\nComputing average Delta from all perturbations...")
    unique_perts = adata.obs["gene"].unique()
    unique_perts = [p for p in unique_perts if p != "NC"]
    print(f"Total perturbations: {len(unique_perts)}")
    
    all_deltas = []
    valid_perts = []
    
    for pert in tqdm(unique_perts, desc="Processing perturbations"):
        pert_indices = np.where(adata.obs["gene"] == pert)[0]
        if len(pert_indices) < 1:
            continue
        
        # 读取扰动细胞的表达数据
        X_pert_data = adata[pert_indices].X
        if hasattr(X_pert_data, "toarray"):
            X_pert = X_pert_data.toarray()
        else:
            X_pert = np.asarray(X_pert_data)
        
        # 计算该扰动的平均表达谱
        pert_mean = np.ravel(np.asarray(np.mean(X_pert, axis=0)))
        
        # 计算 Delta
        delta = pert_mean - control_mean
        all_deltas.append(delta)
        valid_perts.append(pert)
    
    if len(all_deltas) == 0:
        raise ValueError("No valid perturbations found!")
    
    # 计算平均 Delta
    all_deltas = np.array(all_deltas)  # (n_perturbations, n_genes)
    average_delta = np.mean(all_deltas, axis=0)  # (n_genes,)
    
    print(f"\nAverage Delta computed:")
    print(f"  Shape: {average_delta.shape}")
    print(f"  Mean: {average_delta.mean():.6f}")
    print(f"  Std: {average_delta.std():.6f}")
    print(f"  Min: {average_delta.min():.6f}")
    print(f"  Max: {average_delta.max():.6f}")
    
    # 3. 计算平均细胞比例
    print("\nComputing average cell proportions...")
    if df_prop_train is not None:
        # 使用提供的比例数据
        avg_prop = df_prop_train[['pre_adipo', 'adipo', 'lipo', 'other']].mean(axis=0).values
        print(f"Average proportions from file:")
        print(f"  pre_adipo: {avg_prop[0]:.4f}")
        print(f"  adipo: {avg_prop[1]:.4f}")
        print(f"  lipo: {avg_prop[2]:.4f}")
        print(f"  other: {avg_prop[3]:.4f}")
    else:
        # 从数据中计算
        print("Computing proportions from data...")
        all_props = []
        for pert in valid_perts:
            pert_indices = np.where(adata.obs["gene"] == pert)[0]
            pert_obs = adata.obs.iloc[pert_indices]
            prop = np.array([
                pert_obs['pre_adipo'].mean(),
                pert_obs['adipo'].mean(),
                pert_obs['lipo'].mean(),
                pert_obs['other'].mean()
            ])
            all_props.append(prop)
        avg_prop = np.mean(all_props, axis=0)
        print(f"Average proportions from data:")
        print(f"  pre_adipo: {avg_prop[0]:.4f}")
        print(f"  adipo: {avg_prop[1]:.4f}")
        print(f"  lipo: {avg_prop[2]:.4f}")
        print(f"  other: {avg_prop[3]:.4f}")
    
    # 确保比例符合约束条件
    # 1. lipo <= adipo
    if avg_prop[2] > avg_prop[1]:
        avg_prop[2] = avg_prop[1]
    
    # 2. pre_adipo + adipo + other = 1
    sum_main = avg_prop[0] + avg_prop[1] + avg_prop[3]
    if sum_main > 0:
        avg_prop[0] = avg_prop[0] / sum_main
        avg_prop[1] = avg_prop[1] / sum_main
        avg_prop[3] = avg_prop[3] / sum_main
    
    print(f"\nNormalized proportions:")
    print(f"  pre_adipo: {avg_prop[0]:.4f}")
    print(f"  adipo: {avg_prop[1]:.4f}")
    print(f"  lipo: {avg_prop[2]:.4f}")
    print(f"  other: {avg_prop[3]:.4f}")
    print(f"  Sum (pre_adipo + adipo + other): {avg_prop[0] + avg_prop[1] + avg_prop[3]:.4f}")
    print(f"  lipo <= adipo: {avg_prop[2] <= avg_prop[1]}")
    
    # 4. 保存模型数据
    os.makedirs(model_directory_path, exist_ok=True)
    
    # 保存平均 Delta
    joblib.dump(average_delta, os.path.join(model_directory_path, "average_delta.pkl"))
    print(f"\nSaved average_delta.pkl")
    
    # 保存平均比例
    joblib.dump(avg_prop, os.path.join(model_directory_path, "average_proportions.pkl"))
    print(f"Saved average_proportions.pkl")
    
    # 保存控制组均值（用于采样时的参考）
    joblib.dump(control_mean, os.path.join(model_directory_path, "control_mean.pkl"))
    print(f"Saved control_mean.pkl")
    
    # 保存 NC 细胞索引（用于采样）
    joblib.dump(nc_indices, os.path.join(model_directory_path, "nc_indices.pkl"))
    print(f"Saved nc_indices.pkl")
    
    print("\n" + "=" * 80)
    print("Training completed!")
    print("=" * 80)
    
    del adata, X_nc, all_deltas
    gc.collect()
