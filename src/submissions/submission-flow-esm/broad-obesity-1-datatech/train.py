"""
训练脚本：计算基础参数
- 计算 NC 控制组均值
- 计算平均细胞比例
- 保存 NC 细胞索引（用于推理时采样）

注意：prediction_table_fp16_compressed.h5ad 由 prepare_resources.py 预计算，
使用 PM+FlowImp_ode100_pc5_a0.11（α=0.11，ODE 100 步）。
"""

import os
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import joblib
from tqdm import tqdm

from proportion_knn_predictor import ProportionKNNPredictor


def train(
        data_directory_path: str,
        model_directory_path: str,
):
    print("=" * 80)
    print("Training: Compute baseline parameters")
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
        print("Warning: program_proportion.csv not found.")
        df_prop_train = None

    # 1. 提取控制组 (NC) 数据
    print("\nExtracting Control (NC) data...")
    nc_indices = np.where(adata.obs["gene"] == "NC")[0]
    print(f"NC cells: {len(nc_indices)}")

    X_nc_data = adata[nc_indices].X
    if hasattr(X_nc_data, "toarray"):
        X_nc = X_nc_data.toarray()
    else:
        X_nc = np.asarray(X_nc_data)

    control_mean = np.ravel(np.asarray(np.mean(X_nc, axis=0)))
    print(f"Control mean shape: {control_mean.shape}")

    # 2. 计算平均细胞比例
    print("\nComputing average cell proportions...")
    if df_prop_train is not None:
        avg_prop = df_prop_train[['pre_adipo', 'adipo', 'lipo', 'other']].mean(axis=0).values
    else:
        unique_perts = [p for p in adata.obs["gene"].unique() if p != "NC"]
        all_props = []
        for pert in unique_perts:
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

    print(f"Average proportions:")
    print(f"  pre_adipo: {avg_prop[0]:.4f}")
    print(f"  adipo: {avg_prop[1]:.4f}")
    print(f"  lipo: {avg_prop[2]:.4f}")
    print(f"  other: {avg_prop[3]:.4f}")

    # 约束：lipo <= adipo, pre_adipo + adipo + other = 1
    if avg_prop[2] > avg_prop[1]:
        avg_prop[2] = avg_prop[1]
    sum_main = avg_prop[0] + avg_prop[1] + avg_prop[3]
    if sum_main > 0:
        avg_prop[0] /= sum_main
        avg_prop[1] /= sum_main
        avg_prop[3] /= sum_main

    print(f"Normalized proportions:")
    print(f"  pre_adipo: {avg_prop[0]:.4f}, adipo: {avg_prop[1]:.4f}")
    print(f"  lipo: {avg_prop[2]:.4f}, other: {avg_prop[3]:.4f}")

    # 3. 拟合细胞比例 KNN 预测器（仅用 resources 中预提取的紧凑 pkl，不加载大文件）
    scgpt_emb_path = os.path.join(model_directory_path, "scgpt_embeddings.pkl")
    gene2vec_emb_path = os.path.join(model_directory_path, "gene2vec_embeddings.pkl")

    predictor = None
    if df_prop_train is not None:
        df_prop_no_nc = df_prop_train[df_prop_train["gene"] != "NC"]
        train_genes = df_prop_no_nc["gene"].tolist()
        if len(train_genes) < 2:
            print("\nWarning: Too few train genes for ProportionKNNPredictor, skipping")
        elif os.path.exists(scgpt_emb_path):
            emb = joblib.load(scgpt_emb_path)
            predictor = ProportionKNNPredictor(k=15, n_pca=32, embedding_source="scgpt")
            predictor.fit(train_genes, df_prop_train, emb)
            joblib.dump(predictor, os.path.join(model_directory_path, "proportion_knn_predictor.pkl"))
            print(f"\nFitted ProportionKNNPredictor (scGPT, n_pca=32, k=15) on {len(train_genes)} genes")
        elif os.path.exists(gene2vec_emb_path):
            emb = joblib.load(gene2vec_emb_path)
            predictor = ProportionKNNPredictor(k=20, n_pca=32, embedding_source="gene2vec")
            predictor.fit(train_genes, df_prop_train, emb)
            joblib.dump(predictor, os.path.join(model_directory_path, "proportion_knn_predictor.pkl"))
            print(f"\nFitted ProportionKNNPredictor (Gene2vec, n_pca=32, k=20) on {len(train_genes)} genes")
        else:
            print("\nWarning: No scgpt_embeddings.pkl or gene2vec_embeddings.pkl in resources, skipping")

    # 4. 保存
    os.makedirs(model_directory_path, exist_ok=True)
    joblib.dump(avg_prop, os.path.join(model_directory_path, "average_proportions.pkl"))
    joblib.dump(control_mean, os.path.join(model_directory_path, "control_mean.pkl"))
    joblib.dump(nc_indices, os.path.join(model_directory_path, "nc_indices.pkl"))

    print(f"\nSaved: average_proportions.pkl, control_mean.pkl, nc_indices.pkl")
    print("=" * 80)
    print("Training completed!")
    print("=" * 80)

    del adata, X_nc
    gc.collect()
