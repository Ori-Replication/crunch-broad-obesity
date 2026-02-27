"""
训练脚本：SVD + Ridge Delta 预测
- 对 NC 细胞做 TruncatedSVD 得到基因 embedding
- Ridge 回归：gene embedding -> delta (perturbed_mean - control_mean)
- 计算平均细胞比例、保存 NC 细胞索引（用于推理时采样）
- ProportionKNNPredictor（Gene2vec KNN）用于细胞比例预测
"""

import os
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import joblib
from tqdm import tqdm
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge

from proportion_knn_predictor import ProportionKNNPredictor


def train(
        data_directory_path: str,
        model_directory_path: str,
):
    print("=" * 80)
    print("Training: SVD + Ridge Delta Prediction")
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

    # 采样 NC 以控制内存（与 tmp/main.py 一致）
    sample_size = min(len(nc_indices), 20000)
    sampled_nc_indices = np.random.choice(nc_indices, sample_size, replace=False)

    X_nc_data = adata[sampled_nc_indices].X
    if hasattr(X_nc_data, "toarray"):
        X_nc = X_nc_data.toarray()
    else:
        X_nc = np.asarray(X_nc_data)

    control_mean = np.ravel(np.asarray(np.mean(X_nc, axis=0)))
    print(f"Control mean shape: {control_mean.shape}")

    # 2. SVD 基因 embedding + Ridge Delta 回归
    print("\nComputing Gene Embeddings via TruncatedSVD...")
    n_components = 50
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    svd.fit(X_nc)
    gene_embeddings = svd.components_.T  # Shape: (n_genes, 50)
    var_names = adata.var_names.tolist()
    gene_map = {name: gene_embeddings[i] for i, name in enumerate(var_names)}
    print(f"SVD components: {n_components}, gene_map size: {len(gene_map)}")

    print("Preparing Ridge regression training data...")
    X_train_reg = []
    y_train_reg = []
    unique_perts = [p for p in adata.obs["gene"].unique() if p != "NC"]

    for pert in tqdm(unique_perts, desc="Processing perturbations"):
        if pert not in gene_map:
            continue
        pert_indices = np.where(adata.obs["gene"] == pert)[0]
        if len(pert_indices) < 5:
            continue
        idx_to_load = pert_indices[:500]
        X_pert_data = adata[idx_to_load].X
        if hasattr(X_pert_data, "toarray"):
            X_pert = X_pert_data.toarray()
        else:
            X_pert = np.asarray(X_pert_data)
        pert_mean = np.ravel(np.asarray(np.mean(X_pert, axis=0)))
        delta = pert_mean - control_mean
        X_train_reg.append(gene_map[pert])
        y_train_reg.append(delta)

    X_train_reg = np.array(X_train_reg)
    y_train_reg = np.array(y_train_reg)
    print(f"Ridge training: {X_train_reg.shape[0]} perturbations")

    print("Training Ridge regression (alpha=10.0)...")
    shift_model = Ridge(alpha=10.0)
    shift_model.fit(X_train_reg, y_train_reg)

    # 3. 计算平均细胞比例
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

    # 4. 拟合 Gene2vec KNN 细胞比例预测器
    gene2vec_path = os.path.join(model_directory_path, "gene2vec_dim_200_iter_9_w2v.txt")
    script_dir = Path(__file__).resolve().parent
    if not os.path.exists(gene2vec_path):
        # 本地开发时可能 Gene2vec 在 external_repos
        fallback = script_dir.parent.parent.parent.parent / "external_repos" / "Gene2vec" / "pre_trained_emb" / "gene2vec_dim_200_iter_9_w2v.txt"
        if fallback.exists():
            gene2vec_path = str(fallback)

    if df_prop_train is not None and os.path.exists(gene2vec_path):
        df_prop_no_nc = df_prop_train[df_prop_train["gene"] != "NC"]
        train_genes = df_prop_no_nc["gene"].tolist()
        if len(train_genes) >= 2:
            predictor = ProportionKNNPredictor(k=15)
            predictor.fit(train_genes, df_prop_train, gene2vec_path)
            joblib.dump(predictor, os.path.join(model_directory_path, "proportion_knn_predictor.pkl"))
            # 若使用 fallback 路径，复制 Gene2vec 到 model 目录供 infer 使用
            dst_gene2vec = os.path.join(model_directory_path, "gene2vec_dim_200_iter_9_w2v.txt")
            if gene2vec_path != dst_gene2vec:
                import shutil

                shutil.copy2(gene2vec_path, dst_gene2vec)
                print(f"  Copied Gene2vec to {dst_gene2vec}")
            print(f"\nFitted ProportionKNNPredictor (K=15) on {len(train_genes)} genes")
        else:
            print("\nWarning: Too few train genes for ProportionKNNPredictor, skipping")
    else:
        if df_prop_train is None:
            print("\nWarning: program_proportion.csv not found, skipping ProportionKNNPredictor")
        else:
            print(f"\nWarning: Gene2vec not found at {gene2vec_path}, skipping ProportionKNNPredictor")

    # 5. 保存
    os.makedirs(model_directory_path, exist_ok=True)
    joblib.dump(shift_model, os.path.join(model_directory_path, "shift_model.pkl"))
    joblib.dump(gene_map, os.path.join(model_directory_path, "gene_embeddings.pkl"))
    joblib.dump(avg_prop, os.path.join(model_directory_path, "average_proportions.pkl"))
    joblib.dump(control_mean, os.path.join(model_directory_path, "control_mean.pkl"))
    joblib.dump(nc_indices, os.path.join(model_directory_path, "nc_indices.pkl"))

    print(f"\nSaved: shift_model.pkl, gene_embeddings.pkl, average_proportions.pkl, control_mean.pkl, nc_indices.pkl")
    print("=" * 80)
    print("Training completed!")
    print("=" * 80)

    del adata, X_nc, X_train_reg, y_train_reg, gene_map
    gc.collect()
