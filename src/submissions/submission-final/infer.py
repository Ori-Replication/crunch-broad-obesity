"""
推理脚本：SVD + Ridge Delta 预测
- 使用训练好的 shift_model（Ridge）和 gene_embeddings 预测每个扰动的 Delta
- 从 NC 中采样 100 个细胞（均值匹配）
- 对每个扰动的采样细胞加上 Delta 得到预测
- 细胞比例：使用 Gene2vec KNN 预测器（ProportionKNNPredictor），
  找不到 Embedding 的基因用全 0 向量表示；若无预测器则回退到平均比例
"""

import os
import gc
import typing
import numpy as np
import pandas as pd
import scanpy as sc
import anndata
import joblib
import h5py
from tqdm import tqdm


def sample_nc_cells_with_mean_match(X_nc, control_mean, n_samples=100, n_trials=1000):
    """
    从 NC 细胞中采样，使得采样后的均值尽可能接近整体均值
    """
    n_cells = X_nc.shape[0]

    if n_samples >= n_cells:
        return np.arange(n_cells)

    best_indices = None
    best_diff = float('inf')

    for _ in range(n_trials):
        indices = np.random.choice(n_cells, size=n_samples, replace=False)
        sampled_mean = X_nc[indices].mean(axis=0)
        diff = np.linalg.norm(sampled_mean - control_mean)
        if diff < best_diff:
            best_diff = diff
            best_indices = indices

    return best_indices


def infer(
        data_directory_path: str,
        prediction_directory_path: str,
        prediction_h5ad_file_path: str,
        program_proportion_csv_file_path: str,
        model_directory_path: str,
        predict_perturbations: typing.List[str] = None,
        genes_to_predict: typing.List[str] = None,
        cells_per_perturbation: int = 100,
):
    print("=" * 80)
    print("Inference using SVD + Ridge Delta Prediction")
    print("=" * 80)

    # 1. 处理参数
    if genes_to_predict is None:
        genes_file = os.path.join(data_directory_path, "genes_to_predict.txt")
        if os.path.exists(genes_file):
            genes_to_predict = pd.read_csv(genes_file, header=None)[0].tolist()
        else:
            temp_adata = sc.read_h5ad(os.path.join(data_directory_path, "obesity_challenge_1.h5ad"), backed='r')
            genes_to_predict = temp_adata.var_names.tolist()
            del temp_adata

    if predict_perturbations is None:
        pert_file = os.path.join(data_directory_path, "predict_perturbations.txt")
        if os.path.exists(pert_file):
            predict_perturbations = pd.read_csv(pert_file, header=None)[0].tolist()
        else:
            raise FileNotFoundError("predict_perturbations list missing.")

    print(f"Predictions to generate: {len(predict_perturbations)}")
    print(f"Genes to predict: {len(genes_to_predict)}")
    print(f"Cells per perturbation: {cells_per_perturbation}")

    # 2. 加载 SVD + Ridge 模型
    print("\nLoading SVD + Ridge model...")
    shift_model = joblib.load(os.path.join(model_directory_path, "shift_model.pkl"))
    gene_map = joblib.load(os.path.join(model_directory_path, "gene_embeddings.pkl"))
    average_proportions = joblib.load(os.path.join(model_directory_path, "average_proportions.pkl"))
    control_mean = joblib.load(os.path.join(model_directory_path, "control_mean.pkl"))
    nc_indices = joblib.load(os.path.join(model_directory_path, "nc_indices.pkl"))

    default_emb = np.mean(list(gene_map.values()), axis=0)

    # 尝试加载 Gene2vec KNN 细胞比例预测器（找不到 Embedding 的基因用全 0 表示）
    proportion_predictor = None
    gene2vec_path = os.path.join(model_directory_path, "gene2vec_dim_200_iter_9_w2v.txt")
    predictor_path = os.path.join(model_directory_path, "proportion_knn_predictor.pkl")
    if os.path.exists(predictor_path) and os.path.exists(gene2vec_path):
        proportion_predictor = joblib.load(predictor_path)
        print(f"Using ProportionKNNPredictor for cell proportions")
    else:
        print(f"Using average proportions (fallback)")

    # 3. 加载训练数据并准备 NC 细胞
    print("\nLoading NC cells from training data...")
    temp = sc.read_h5ad(os.path.join(data_directory_path, "obesity_challenge_1.h5ad"), backed='r')
    all_var_names = temp.var_names.tolist()
    gene_idx_map = {g: i for i, g in enumerate(all_var_names)}

    common_genes = [g for g in genes_to_predict if g in gene_idx_map]
    if len(common_genes) != len(genes_to_predict):
        missing = [g for g in genes_to_predict if g not in gene_idx_map]
        print(f"Warning: {len(missing)} genes not in training data")
        print(f"Using {len(common_genes)} common genes")
        genes_to_predict = common_genes

    target_indices = [gene_idx_map[g] for g in genes_to_predict]

    # 读取 NC 细胞表达数据（仅目标基因）
    print("Reading NC expression data (target genes only)...")
    X_nc_full = temp[nc_indices].X
    if hasattr(X_nc_full, "toarray"):
        X_nc_full = X_nc_full.toarray()
    X_nc = X_nc_full[:, target_indices].astype(np.float32)

    control_mean_subset = np.asarray(control_mean)[target_indices].astype(np.float32)
    print(f"NC cells shape: {X_nc.shape}")

    # 4. 使用 SVD+Ridge 预测每个扰动的 Delta
    print("\nPredicting Deltas via SVD + Ridge...")
    if proportion_predictor is not None:
        prop_pred_df = proportion_predictor.predict(predict_perturbations, gene2vec_path)
        prop_by_pert = {row["gene"]: row[["pre_adipo", "adipo", "lipo", "other"]].values for _, row in prop_pred_df.iterrows()}
    else:
        prop_by_pert = {p: average_proportions.copy() for p in predict_perturbations}

    perturbation_deltas = {}
    for pert in tqdm(predict_perturbations, desc="Predicting Deltas"):
        emb = gene_map.get(pert, default_emb).reshape(1, -1)
        pred_delta_full = shift_model.predict(emb).flatten()
        pred_delta_sub = pred_delta_full[target_indices].astype(np.float32)
        perturbation_deltas[pert] = pred_delta_sub

    print(f"Computed Deltas for {len(perturbation_deltas)} perturbations")

    # 5. 采样 NC 细胞
    print(f"\nSampling {cells_per_perturbation} NC cells with mean matching...")
    sampled_nc_indices = sample_nc_cells_with_mean_match(
        X_nc, control_mean_subset,
        n_samples=cells_per_perturbation,
        n_trials=1000
    )

    sampled_mean = X_nc[sampled_nc_indices].mean(axis=0)
    mean_diff = np.linalg.norm(sampled_mean - control_mean_subset)
    print(f"Sampled mean difference from control mean: {mean_diff:.6f}")

    base_cells = X_nc[sampled_nc_indices].copy()
    print(f"Base cells shape: {base_cells.shape}")

    del X_nc
    gc.collect()

    # 6. 生成预测
    n_perturbations = len(predict_perturbations)
    n_cells = n_perturbations * cells_per_perturbation
    n_genes = len(genes_to_predict)

    print(f"\nGenerating predictions:")
    print(f"  Total cells: {n_cells}")
    print(f"  Total genes: {n_genes}")

    if os.path.exists(prediction_h5ad_file_path):
        os.remove(prediction_h5ad_file_path)

    temporary_prediction_path = os.path.join(prediction_directory_path, "prediction_matrix_temp.h5")
    if os.path.exists(temporary_prediction_path):
        os.remove(temporary_prediction_path)

    print("Using HDF5 low-memory mode...")

    obs_genes = []
    prop_preds = []

    with h5py.File(temporary_prediction_path, "w") as f:
        batch_size = cells_per_perturbation
        dset = f.create_dataset(
            "X",
            shape=(n_cells, n_genes),
            dtype="float32",
            chunks=(batch_size, n_genes),
        )

        for i, pert in enumerate(tqdm(predict_perturbations, desc="Predicting")):
            pert_delta = perturbation_deltas[pert]
            pert_cells = base_cells + pert_delta
            pert_cells = np.clip(pert_cells, 0, None)
            pert_cells = pert_cells.astype(np.float32)

            start_idx = i * cells_per_perturbation
            end_idx = start_idx + cells_per_perturbation
            dset[start_idx:end_idx] = pert_cells

            obs_genes.extend([pert] * cells_per_perturbation)
            prop_preds.append(prop_by_pert[pert].copy())

            if i % 1000 == 0:
                gc.collect()

    print("Finished writing prediction matrix to HDF5.")

    del base_cells
    gc.collect()

    # 7. 创建 AnnData 对象
    print("Creating AnnData object from HDF5...")
    obs = pd.DataFrame({"gene": obs_genes})
    var_subset = temp.var.loc[genes_to_predict].copy()

    with h5py.File(temporary_prediction_path, "r") as f:
        pred_adata = anndata.AnnData(
            X=np.zeros((n_cells, n_genes), dtype=np.float32),
            obs=obs,
            var=var_subset
        )
        pred_adata.X = f["X"]

        print("Saving prediction to h5ad file...")
        pred_adata.write_h5ad(prediction_h5ad_file_path)

    if os.path.exists(temporary_prediction_path):
        os.remove(temporary_prediction_path)
        print("Cleaned up temporary HDF5 file.")

    # 8. 保存比例预测
    df_props = pd.DataFrame(prop_preds, columns=['pre_adipo', 'adipo', 'lipo', 'other'])
    df_props['gene'] = predict_perturbations
    df_props = df_props[["gene", "pre_adipo", "adipo", "lipo", "other"]]
    df_props.to_csv(program_proportion_csv_file_path, index=False)

    print("\n" + "=" * 80)
    print("Inference completed!")
    print("=" * 80)

    del temp
    gc.collect()
