"""
推理脚本：使用 Perturbmean baseline 进行预测
- 从 NC 中采样 100 个细胞（保证采样后均值接近整体均值）
- 加上平均 Delta 得到预测
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
    print("Perturbmean Baseline Inference")
    print("=" * 80)
    
    # 1. 加载参数
    if genes_to_predict is None:
        genes_file = os.path.join(data_directory_path, "genes_to_predict.txt")
        if os.path.exists(genes_file):
            genes_to_predict = pd.read_csv(genes_file, header=None)[0].tolist()
        else:
            temp = sc.read_h5ad(os.path.join(data_directory_path, "obesity_challenge_1.h5ad"), backed='r')
            genes_to_predict = temp.var_names.tolist()
            del temp
    
    if predict_perturbations is None:
        pert_file = os.path.join(data_directory_path, "predict_perturbations.txt")
        if os.path.exists(pert_file):
            predict_perturbations = pd.read_csv(pert_file, header=None)[0].tolist()
        else:
            raise FileNotFoundError("predict_perturbations list missing.")
    
    print(f"Predictions to generate: {len(predict_perturbations)}")
    print(f"Genes to predict: {len(genes_to_predict)}")
    print(f"Cells per perturbation: {cells_per_perturbation}")
    
    # 2. 加载模型数据
    print("\nLoading model data...")
    average_delta = joblib.load(os.path.join(model_directory_path, "average_delta.pkl"))
    average_proportions = joblib.load(os.path.join(model_directory_path, "average_proportions.pkl"))
    control_mean = joblib.load(os.path.join(model_directory_path, "control_mean.pkl"))
    nc_indices = joblib.load(os.path.join(model_directory_path, "nc_indices.pkl"))
    
    print(f"Average delta shape: {average_delta.shape}")
    print(f"Average proportions: {average_proportions}")
    
    # 3. 加载训练数据并准备 NC 细胞
    print("\nLoading NC cells from training data...")
    temp = sc.read_h5ad(os.path.join(data_directory_path, "obesity_challenge_1.h5ad"), backed='r')
    all_var_names = temp.var_names.tolist()
    gene_idx_map = {g: i for i, g in enumerate(all_var_names)}
    target_indices = [gene_idx_map[g] for g in genes_to_predict if g in gene_idx_map]
    
    # 读取 NC 细胞的表达数据（仅目标基因）
    print("Reading NC expression data (target genes only)...")
    X_nc_full = temp[nc_indices].X
    if hasattr(X_nc_full, "toarray"):
        X_nc_full = X_nc_full.toarray()
    X_nc = X_nc_full[:, target_indices].astype(np.float32)
    
    # 获取控制组均值（仅目标基因）
    control_mean_subset = control_mean[target_indices].astype(np.float32)
    
    # 获取平均 Delta（仅目标基因）
    average_delta_subset = average_delta[target_indices].astype(np.float32)
    
    print(f"NC cells shape: {X_nc.shape}")
    print(f"Control mean (subset) shape: {control_mean_subset.shape}")
    print(f"Average delta (subset) shape: {average_delta_subset.shape}")
    
    # 4. 采样 NC 细胞（只采样一次，然后复制给所有扰动）
    print(f"\nSampling {cells_per_perturbation} NC cells with mean matching (once for all perturbations)...")
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
    
    # 获取采样细胞的表达谱并加上 Delta（只计算一次）
    base_cells = X_nc[sampled_nc_indices].copy()  # (cells_per_perturbation, n_genes)
    generated_cells_template = base_cells + average_delta_subset  # (cells_per_perturbation, n_genes)
    generated_cells_template = np.clip(generated_cells_template, 0, None)  # 修正负值
    generated_cells_template = generated_cells_template.astype(np.float32)
    
    print(f"Generated template cells shape: {generated_cells_template.shape}")
    
    # 释放内存
    del base_cells, X_nc
    gc.collect()
    
    # 5. 准备输出
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
    
    # 6. 生成预测（复制模板给每个扰动）
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
            # 直接复制模板细胞给每个扰动（不需要重新采样和计算）
            start_idx = i * cells_per_perturbation
            end_idx = start_idx + cells_per_perturbation
            dset[start_idx:end_idx] = generated_cells_template
            
            obs_genes.extend([pert] * cells_per_perturbation)
            
            # 使用平均比例（已经满足约束条件）
            prop_preds.append(average_proportions.copy())
            
            if i % 1000 == 0:
                gc.collect()
    
    print("Finished writing prediction matrix to HDF5.")
    
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
    
    # 清理临时文件
    if os.path.exists(temporary_prediction_path):
        os.remove(temporary_prediction_path)
        print("Cleaned up temporary HDF5 file.")
    
    # 8. 保存比例预测
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
