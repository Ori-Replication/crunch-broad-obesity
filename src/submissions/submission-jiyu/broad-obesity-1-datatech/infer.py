"""
推理脚本：使用预预测结果进行预测
- 从预预测文件中读取每个扰动的均值细胞
- 计算每个扰动相对于 NC 的 Delta
- 从 NC 中采样 100 个细胞（保证采样后均值接近整体均值）
- 对每个扰动的采样细胞加上对应的 Delta 得到预测
- 使用平均细胞比例
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
    
    参数:
        X_nc: NC 细胞的表达矩阵 (n_cells, n_genes)
        control_mean: NC 整体的平均表达谱 (n_genes,)
        n_samples: 需要采样的细胞数量
        n_trials: 尝试次数，选择均值最接近的样本
    
    返回:
        selected_indices: 选中的细胞索引
    """
    n_cells = X_nc.shape[0]
    
    if n_samples >= n_cells:
        # 如果需要的样本数大于等于总数，直接返回所有索引
        return np.arange(n_cells)
    
    best_indices = None
    best_diff = float('inf')
    
    for _ in range(n_trials):
        # 随机采样
        indices = np.random.choice(n_cells, size=n_samples, replace=False)
        sampled_mean = X_nc[indices].mean(axis=0)
        
        # 计算与整体均值的差异（L2 距离）
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
    print("Inference using Pre-predicted Results")
    print("=" * 80)
    
    # 1. 加载预预测结果文件
    prediction_table_path = os.path.join(model_directory_path, "prediction_table_fp16_compressed.h5ad")
    if not os.path.exists(prediction_table_path):
        raise FileNotFoundError(f"Pre-prediction file not found: {prediction_table_path}")
    
    print(f"\nLoading pre-prediction file: {prediction_table_path}")
    pred_table = sc.read_h5ad(prediction_table_path, backed='r')
    print(f"Pre-prediction shape: {pred_table.shape}")
    print(f"Pre-prediction obs columns: {pred_table.obs.columns.tolist()}")
    
    # 获取预预测文件中的扰动列表
    if 'gene' not in pred_table.obs.columns:
        raise ValueError("Pre-prediction file must have 'gene' column in obs")
    
    pred_perturbations = pred_table.obs['gene'].unique().tolist()
    print(f"Found {len(pred_perturbations)} perturbations in pre-prediction file")
    
    # 2. 加载参数
    if genes_to_predict is None:
        genes_file = os.path.join(data_directory_path, "genes_to_predict.txt")
        if os.path.exists(genes_file):
            genes_to_predict = pd.read_csv(genes_file, header=None)[0].tolist()
        else:
            # 从预预测文件中获取基因列表（需要确保也在训练数据中存在）
            genes_to_predict = pred_table.var_names.tolist()
    
    if predict_perturbations is None:
        pert_file = os.path.join(data_directory_path, "predict_perturbations.txt")
        if os.path.exists(pert_file):
            predict_perturbations = pd.read_csv(pert_file, header=None)[0].tolist()
        else:
            # 如果没有指定，使用预预测文件中的所有扰动
            predict_perturbations = pred_perturbations
    
    # 确保所有需要预测的扰动都在预预测文件中
    missing_perts = [p for p in predict_perturbations if p not in pred_perturbations]
    if missing_perts:
        raise ValueError(f"Missing perturbations in pre-prediction file: {missing_perts}")
    
    print(f"Predictions to generate: {len(predict_perturbations)}")
    print(f"Genes to predict: {len(genes_to_predict)}")
    print(f"Cells per perturbation: {cells_per_perturbation}")
    
    # 3. 加载模型数据（用于 NC 细胞和比例）
    print("\nLoading model data...")
    average_proportions = joblib.load(os.path.join(model_directory_path, "average_proportions.pkl"))
    control_mean = joblib.load(os.path.join(model_directory_path, "control_mean.pkl"))
    nc_indices = joblib.load(os.path.join(model_directory_path, "nc_indices.pkl"))
    
    print(f"Average proportions: {average_proportions}")
    
    # 4. 加载训练数据并准备 NC 细胞
    print("\nLoading NC cells from training data...")
    temp = sc.read_h5ad(os.path.join(data_directory_path, "obesity_challenge_1.h5ad"), backed='r')
    all_var_names = temp.var_names.tolist()
    gene_idx_map = {g: i for i, g in enumerate(all_var_names)}
    
    # 确保基因在训练数据和预预测文件中都存在
    pred_gene_set = set(pred_table.var_names)
    train_gene_set = set(all_var_names)
    common_genes = [g for g in genes_to_predict if g in gene_idx_map and g in pred_gene_set]
    
    if len(common_genes) != len(genes_to_predict):
        missing_in_train = [g for g in genes_to_predict if g not in train_gene_set]
        missing_in_pred = [g for g in genes_to_predict if g not in pred_gene_set]
        if missing_in_train:
            print(f"Warning: {len(missing_in_train)} genes not found in training data: {missing_in_train[:5]}...")
        if missing_in_pred:
            print(f"Warning: {len(missing_in_pred)} genes not found in pre-prediction file: {missing_in_pred[:5]}...")
        print(f"Using {len(common_genes)} common genes for prediction")
        genes_to_predict = common_genes
    
    target_indices = [gene_idx_map[g] for g in genes_to_predict]
    
    # 读取 NC 细胞的表达数据（仅目标基因）
    print("Reading NC expression data (target genes only)...")
    X_nc_full = temp[nc_indices].X
    if hasattr(X_nc_full, "toarray"):
        X_nc_full = X_nc_full.toarray()
    X_nc = X_nc_full[:, target_indices].astype(np.float32)
    
    # 获取控制组均值（仅目标基因）
    control_mean_subset = control_mean[target_indices].astype(np.float32)
    
    print(f"NC cells shape: {X_nc.shape}")
    print(f"Control mean (subset) shape: {control_mean_subset.shape}")
    
    # 5. 从预预测文件中提取每个扰动的均值细胞并计算 Delta
    print("\nExtracting perturbation means and computing Deltas from pre-prediction file...")
    
    # 创建基因索引映射（预预测文件中的基因顺序，确保与 genes_to_predict 顺序一致）
    pred_gene_idx_map = {g: i for i, g in enumerate(pred_table.var_names)}
    pred_target_indices = [pred_gene_idx_map[g] for g in genes_to_predict]
    
    # 存储每个扰动的 Delta
    perturbation_deltas = {}
    
    for pert in tqdm(predict_perturbations, desc="Computing Deltas"):
        # 从预预测文件中提取该扰动的均值细胞
        pert_mask = pred_table.obs['gene'] == pert
        if pert_mask.sum() == 0:
            raise ValueError(f"No cells found for perturbation: {pert}")
        
        # 提取均值细胞（应该只有1个）
        pert_mean_cell = pred_table[pert_mask].X
        if hasattr(pert_mean_cell, "toarray"):
            pert_mean_cell = pert_mean_cell.toarray()
        pert_mean_cell = np.asarray(pert_mean_cell)
        
        if pert_mean_cell.shape[0] != 1:
            print(f"Warning: Expected 1 cell for {pert}, found {pert_mean_cell.shape[0]}. Using mean.")
            pert_mean_cell = pert_mean_cell.mean(axis=0, keepdims=True)
        
        # 提取目标基因的表达（确保顺序与 genes_to_predict 一致）
        pert_mean_subset = pert_mean_cell[0, pred_target_indices].astype(np.float32)
        
        # 计算相对于 NC 的 Delta
        delta = pert_mean_subset - control_mean_subset
        perturbation_deltas[pert] = delta
    
    print(f"Computed Deltas for {len(perturbation_deltas)} perturbations")
    
    # 6. 采样 NC 细胞（只采样一次，然后对每个扰动应用不同的 Delta）
    print(f"\nSampling {cells_per_perturbation} NC cells with mean matching...")
    sampled_nc_indices = sample_nc_cells_with_mean_match(
        X_nc, control_mean_subset, 
        n_samples=cells_per_perturbation, 
        n_trials=1000
    )
    
    # 验证采样后的均值
    sampled_mean = X_nc[sampled_nc_indices].mean(axis=0)
    mean_diff = np.linalg.norm(sampled_mean - control_mean_subset)
    print(f"Sampled mean difference from control mean: {mean_diff:.6f}")
    print(f"Relative difference: {mean_diff / (np.linalg.norm(control_mean_subset) + 1e-8):.6f}")
    
    # 获取采样细胞的表达谱（作为基础）
    base_cells = X_nc[sampled_nc_indices].copy()  # (cells_per_perturbation, n_genes)
    
    print(f"Base cells shape: {base_cells.shape}")
    
    # 释放内存
    del X_nc, pred_table
    gc.collect()
    
    # 6. 准备输出
    n_perturbations = len(predict_perturbations)
    n_cells = n_perturbations * cells_per_perturbation
    n_genes = len(genes_to_predict)
    
    print(f"\nGenerating predictions:")
    print(f"  Total cells: {n_cells}")
    print(f"  Total genes: {n_genes}")
    
    # 清理旧的预测文件
    if os.path.exists(prediction_h5ad_file_path):
        os.remove(prediction_h5ad_file_path)
    
    temporary_prediction_path = os.path.join(prediction_directory_path, "prediction_matrix_temp.h5")
    if os.path.exists(temporary_prediction_path):
        os.remove(temporary_prediction_path)
    
    print("Using HDF5 low-memory mode...")
    
    # 7. 生成预测（对每个扰动应用对应的 Delta）
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
            # 获取该扰动的 Delta
            pert_delta = perturbation_deltas[pert]
            
            # 对基础细胞应用该扰动的 Delta
            pert_cells = base_cells + pert_delta  # (cells_per_perturbation, n_genes)
            pert_cells = np.clip(pert_cells, 0, None)  # 修正负值
            pert_cells = pert_cells.astype(np.float32)
            
            # 写入预测结果
            start_idx = i * cells_per_perturbation
            end_idx = start_idx + cells_per_perturbation
            dset[start_idx:end_idx] = pert_cells
            
            obs_genes.extend([pert] * cells_per_perturbation)
            
            # 使用平均比例（已经满足约束条件）
            prop_preds.append(average_proportions.copy())
            
            if i % 1000 == 0:
                gc.collect()
    
    print("Finished writing prediction matrix to HDF5.")
    
    # 释放基础细胞内存
    del base_cells
    gc.collect()
    
    # 8. 创建 AnnData 对象
    print("Creating AnnData object from HDF5...")
    
    obs = pd.DataFrame({"gene": obs_genes})
    # 从训练数据中获取 var 信息（genes_to_predict 已经确保在训练数据中存在）
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
    
    # 清理临时文件
    if os.path.exists(temporary_prediction_path):
        os.remove(temporary_prediction_path)
        print("Cleaned up temporary HDF5 file.")
    
    # 9. 保存比例预测
    df_props = pd.DataFrame(prop_preds, columns=['pre_adipo', 'adipo', 'lipo', 'other'])
    df_props['gene'] = predict_perturbations
    
    # 调整列顺序
    df_props = df_props[["gene", "pre_adipo", "adipo", "lipo", "other"]]
    df_props.to_csv(program_proportion_csv_file_path, index=False)
    
    print("\n" + "=" * 80)
    print("Inference completed!")
    print("=" * 80)
    
    del temp
    gc.collect()
