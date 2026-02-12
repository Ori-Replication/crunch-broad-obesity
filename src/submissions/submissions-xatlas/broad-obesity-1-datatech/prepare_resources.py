"""
预计算脚本：使用 PCA_HEK_pc5_alpha100 方法生成预测表
本脚本在本地运行，生成 resources/ 中的所有文件

方法：
1. 对竞赛训练 delta 做 PCA（5 个主成分）
2. 对 xatlas HEK293T delta 做 PCA（30 个主成分）
3. 训练 Ridge 回归：xatlas PCA features -> competition PC scores（alpha=100）
4. 对所有 2863 个预测目标，使用 Ridge 预测竞赛 PC scores，重建 delta
5. 不在 xatlas 中的扰动使用 PerturbMean（平均 delta）
6. 保存预测表

运行方式：
    cd CrunchDAO-obesity
    conda activate broad
    python src/submissions/submissions-xatlas/broad-obesity-1-datatech/prepare_resources.py
"""

import os
import gc
import sys
import numpy as np
import pandas as pd
import scanpy as sc
import anndata
import joblib
from pathlib import Path
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr

# 路径配置
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "original_data" / "default"
XATLAS_DIR = PROJECT_ROOT / "data" / "external_data" / "xatlas" / "processed"
RESOURCES_DIR = SCRIPT_DIR / "resources"
CACHE_DIR = PROJECT_ROOT / "eda" / "xatlas_transfer" / "cache"

RESOURCES_DIR.mkdir(exist_ok=True, parents=True)


def to_array(x):
    if isinstance(x, np.ndarray): return x
    if hasattr(x, 'toarray'): return x.toarray()
    return np.array(x)


def load_or_aggregate_xatlas(target_perts, comp_genes, cell_line="HEK293T"):
    """聚合 xatlas delta，使用缓存"""
    cache_file = CACHE_DIR / f"xatlas_deltas_{cell_line}_full.parquet"

    if cache_file.exists():
        print(f"  Using cache: {cache_file.name}", flush=True)
        return pd.read_parquet(cache_file)

    print(f"  Aggregating {cell_line} batches...", flush=True)
    batch_dir = XATLAS_DIR / "by_batch"
    batch_files = sorted([f for f in os.listdir(batch_dir)
                          if f.startswith(cell_line + "_") and f.endswith("_aggregated.parquet")])
    print(f"  Batches: {len(batch_files)}", flush=True)

    first_batch = pd.read_parquet(batch_dir / batch_files[0])
    overlapping_genes = sorted(list(set(comp_genes) & set(first_batch.columns)))
    print(f"  Overlapping genes: {len(overlapping_genes)}/{len(comp_genes)}", flush=True)
    del first_batch; gc.collect()

    target_set = set(target_perts) | {"Non-Targeting"}
    delta_sum, delta_count = {}, {}

    for fn in tqdm(batch_files, desc=f"  Agg {cell_line}"):
        try:
            b = pd.read_parquet(batch_dir / fn, columns=overlapping_genes)
            if "Non-Targeting" not in b.index:
                del b; gc.collect(); continue
            ctrl = b.loc["Non-Targeting"].values
            for p in set(b.index) & target_set - {"Non-Targeting"}:
                d = b.loc[p].values - ctrl
                if p not in delta_sum:
                    delta_sum[p] = np.zeros(len(overlapping_genes), dtype=np.float64)
                    delta_count[p] = 0
                delta_sum[p] += d
                delta_count[p] += 1
            del b; gc.collect()
        except:
            continue

    avg = {p: delta_sum[p] / delta_count[p] for p in delta_sum if delta_count[p] > 0}
    xatlas_deltas = pd.DataFrame(avg, index=overlapping_genes).T
    xatlas_deltas.to_parquet(cache_file)
    print(f"  Done: {len(xatlas_deltas)} perts, {xatlas_deltas.shape[1]} genes", flush=True)
    return xatlas_deltas


def main():
    print("=" * 80)
    print("Prepare Resources: PCA_HEK_pc5_alpha100 Method")
    print("=" * 80, flush=True)

    # ============================================
    # 1. 加载竞赛数据
    # ============================================
    print("\n[1] Loading competition data...", flush=True)
    adata = sc.read_h5ad(DATA_DIR / "obesity_challenge_1.h5ad", backed='r')
    gene_names = adata.var.index.tolist()
    n_genes = len(gene_names)

    # NC 数据
    nc_indices = np.where(adata.obs['gene'] == 'NC')[0]
    X_nc = to_array(adata[nc_indices].X)
    control_mean = X_nc.mean(axis=0)
    print(f"  NC cells: {len(nc_indices)}, genes: {n_genes}", flush=True)

    # 训练扰动 delta
    unique_perts = [p for p in adata.obs['gene'].cat.categories if p != 'NC']
    print(f"  Training perturbations: {len(unique_perts)}", flush=True)

    train_deltas = {}
    for pert in tqdm(unique_perts, desc="  Computing train deltas"):
        mask = adata.obs['gene'] == pert
        if mask.sum() > 0:
            X_pert = to_array(adata[mask].X)
            pert_mean = X_pert.mean(axis=0)
            train_deltas[pert] = pert_mean - control_mean

    train_perts = list(train_deltas.keys())
    train_deltas_np = np.array([train_deltas[p] for p in train_perts])
    print(f"  Valid training perturbations: {len(train_perts)}", flush=True)

    # 平均 delta（PerturbMean fallback）
    average_delta = train_deltas_np.mean(axis=0)

    # 加载预测目标
    predict_perts_file = DATA_DIR / "predict_perturbations.txt"
    predict_perturbations = pd.read_csv(predict_perts_file, header=None)[0].tolist()
    print(f"  Prediction targets: {len(predict_perturbations)}", flush=True)

    # 细胞比例
    proportion_path = DATA_DIR / "program_proportion.csv"
    df_prop = pd.read_csv(proportion_path)
    avg_prop = df_prop[['pre_adipo', 'adipo', 'lipo', 'other']].mean(axis=0).values
    # 约束：lipo <= adipo, pre_adipo + adipo + other = 1
    if avg_prop[2] > avg_prop[1]:
        avg_prop[2] = avg_prop[1]
    sm = avg_prop[0] + avg_prop[1] + avg_prop[3]
    if sm > 0:
        avg_prop[0] /= sm; avg_prop[1] /= sm; avg_prop[3] /= sm
    print(f"  Avg proportions: pre={avg_prop[0]:.4f} adipo={avg_prop[1]:.4f} "
          f"lipo={avg_prop[2]:.4f} other={avg_prop[3]:.4f}", flush=True)

    # ============================================
    # 2. 聚合 xatlas HEK293T
    # ============================================
    print("\n[2] Loading xatlas HEK293T data...", flush=True)
    all_target_perts = list(set(train_perts + predict_perturbations))
    xatlas_deltas = load_or_aggregate_xatlas(all_target_perts, gene_names, "HEK293T")
    xatlas_genes = xatlas_deltas.columns.tolist()

    train_in_xatlas = [p for p in train_perts if p in xatlas_deltas.index]
    pred_in_xatlas = [p for p in predict_perturbations if p in xatlas_deltas.index]
    print(f"  Train in xatlas: {len(train_in_xatlas)}/{len(train_perts)}", flush=True)
    print(f"  Predict in xatlas: {len(pred_in_xatlas)}/{len(predict_perturbations)}", flush=True)

    # ============================================
    # 3. 训练 PCA + Ridge 模型
    # ============================================
    print("\n[3] Training PCA + Ridge model...", flush=True)

    # PCA on training deltas (5 components)
    n_comp = 5
    pca = PCA(n_components=n_comp)
    train_scores = pca.fit_transform(train_deltas_np)
    print(f"  PCA explained variance: {pca.explained_variance_ratio_.sum():.4f}", flush=True)

    # xatlas features for training perturbations
    valid_train_idx = [i for i, p in enumerate(train_perts) if p in xatlas_deltas.index]
    valid_train_perts = [train_perts[i] for i in valid_train_idx]
    xatlas_train = xatlas_deltas.loc[valid_train_perts].values

    # PCA on xatlas features
    n_xatlas_pc = min(30, len(valid_train_idx) - 1, xatlas_train.shape[1])
    xatlas_pca = PCA(n_components=n_xatlas_pc)
    X_train = xatlas_pca.fit_transform(xatlas_train)
    print(f"  xatlas PCA components: {n_xatlas_pc}", flush=True)

    # Ridge regression
    ridge = Ridge(alpha=100.0, fit_intercept=True)
    Y_train = train_scores[valid_train_idx]
    ridge.fit(X_train, Y_train)

    # 内部评估
    Y_pred_train = ridge.predict(X_train)
    train_reconstructed = pca.inverse_transform(Y_pred_train)
    train_pearsons = []
    for i, idx in enumerate(valid_train_idx):
        corr, _ = pearsonr(train_reconstructed[i], train_deltas_np[idx])
        if not np.isnan(corr):
            train_pearsons.append(corr)
    print(f"  Training Pearson (resubstitution): {np.mean(train_pearsons):.4f}", flush=True)

    # ============================================
    # 4. 生成预测表
    # ============================================
    print("\n[4] Generating prediction table...", flush=True)

    prediction_means = {}  # pert -> predicted mean expression profile

    n_pca_pred = 0
    n_fallback = 0

    for pert in tqdm(predict_perturbations, desc="  Predicting"):
        if pert in xatlas_deltas.index:
            # PCA transfer
            x = xatlas_deltas.loc[pert].values.reshape(1, -1)
            x_pca = xatlas_pca.transform(x)
            scores = ridge.predict(x_pca)
            delta = pca.inverse_transform(scores).flatten()
            prediction_means[pert] = (control_mean + delta).astype(np.float32)
            n_pca_pred += 1
        else:
            # Fallback to PerturbMean
            prediction_means[pert] = (control_mean + average_delta).astype(np.float32)
            n_fallback += 1

    print(f"  PCA predictions: {n_pca_pred}", flush=True)
    print(f"  Fallback (PerturbMean): {n_fallback}", flush=True)

    # 创建 AnnData 预测表
    pred_matrix = np.array([prediction_means[p] for p in predict_perturbations])
    pred_obs = pd.DataFrame({'gene': predict_perturbations})
    pred_var = pd.DataFrame(index=gene_names)
    pred_adata = anndata.AnnData(X=pred_matrix, obs=pred_obs, var=pred_var)

    # 保存为 fp16 压缩
    pred_adata.X = pred_adata.X.astype(np.float16)
    pred_path = RESOURCES_DIR / "prediction_table_fp16_compressed.h5ad"
    pred_adata.write_h5ad(pred_path)
    print(f"  Saved: {pred_path} ({os.path.getsize(pred_path) / 1e6:.1f} MB)", flush=True)

    # ============================================
    # 5. 保存其他资源
    # ============================================
    print("\n[5] Saving resources...", flush=True)

    joblib.dump(average_delta, RESOURCES_DIR / "average_delta.pkl")
    joblib.dump(avg_prop, RESOURCES_DIR / "average_proportions.pkl")
    joblib.dump(control_mean, RESOURCES_DIR / "control_mean.pkl")
    joblib.dump(nc_indices, RESOURCES_DIR / "nc_indices.pkl")

    print(f"  Saved: average_delta.pkl, average_proportions.pkl, control_mean.pkl, nc_indices.pkl", flush=True)

    # ============================================
    # 6. 验证
    # ============================================
    print("\n[6] Validation...", flush=True)
    pred_check = sc.read_h5ad(pred_path)
    print(f"  Prediction table shape: {pred_check.shape}", flush=True)
    print(f"  Perturbations: {len(pred_check.obs['gene'].unique())}", flush=True)
    print(f"  X dtype: {pred_check.X.dtype}", flush=True)
    print(f"  X range: [{pred_check.X.min():.4f}, {pred_check.X.max():.4f}]", flush=True)

    print("\n" + "=" * 80)
    print("Resources prepared successfully!")
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
