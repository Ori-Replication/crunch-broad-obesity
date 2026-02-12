import os
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.stats
import scipy.sparse as scipy_sparse
from tqdm import tqdm


def compute_pearson_delta(
        gtruth_X,  # 真实细胞表达矩阵 (n_cells, n_genes)
        pred_X,  # 预测细胞表达矩阵 (n_cells, n_genes)
        perturbed_centroid,  # 训练集中所有扰动细胞的均值 (n_genes,)
):
    """计算Pearson Delta指标"""
    gtruth_mean = gtruth_X.mean(axis=0)
    pred_mean = pred_X.mean(axis=0)

    gtruth_effect = gtruth_mean - perturbed_centroid
    pred_effect = pred_mean - perturbed_centroid

    # 计算皮尔逊相关系数
    corr, _ = scipy.stats.pearsonr(gtruth_effect, pred_effect)
    return corr


def gaussian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=2326):
    """计算高斯核函数"""
    n_samples = int(source.shape[0]) + int(target.shape[0])
    total = np.concatenate([source, target], axis=0)

    total0 = np.expand_dims(total, axis=0)
    total1 = np.expand_dims(total, axis=1)
    L2_distance = np.sum((total0 - total1) ** 2, axis=2)

    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = np.sum(L2_distance) / (n_samples ** 2 - n_samples)

    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [np.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]

    return sum(kernel_val)


def compute_mmd(gtruth_X, pred_X):
    """计算MMD指标"""
    kernel = gaussian_kernel(gtruth_X, pred_X)
    n = gtruth_X.shape[0]

    XX = kernel[:n, :n]
    YY = kernel[n:, n:]
    XY = kernel[:n, n:]
    YX = kernel[n:, :n]

    mmd_val = np.sum(XX + YY - XY - YX)
    return mmd_val / n ** 2


def compute_l1_distance(true_proportions, pred_proportions):
    """计算程序水平的L1距离"""
    unique_genes = true_proportions["gene"].unique()
    all_l1 = []

    for gene in unique_genes:
        true_row = true_proportions[true_proportions["gene"] == gene].iloc[0]
        pred_row = pred_proportions[pred_proportions["gene"] == gene].iloc[0]

        # 三个主要比例的L1损失
        l1_three = (
                np.abs(true_row["pre_adipo"] - pred_row["pre_adipo"]) +
                np.abs(true_row["adipo"] - pred_row["adipo"]) +
                np.abs(true_row["other"] - pred_row["other"])
        )

        # lipo/adipo的条件概率
        epsilon = 1e-20
        true_ratio = true_row["lipo"] / (true_row["adipo"] + epsilon)
        pred_ratio = pred_row["lipo"] / (pred_row["adipo"] + epsilon)
        l1_ratio = np.abs(true_ratio - pred_ratio)

        # 加权平均
        avg_l1 = 0.75 * l1_three + 0.25 * l1_ratio
        all_l1.append(avg_l1)

    return np.mean(all_l1)


def evaluate_predictions(
        prediction_h5ad_file_path: str,
        gtruth_h5ad_file_path: str,
        gtruth_proportion_csv_file_path: str,
        train_h5ad_file_path: str,
        n_genes_for_metric: int = 1000,
        random_seed: int = 42,
):
    """
    评估预测结果的多个指标

    参数:
        prediction_h5ad_file_path: 预测的AnnData文件路径
        gtruth_h5ad_file_path: 真实的测试集AnnData文件路径
        gtruth_proportion_csv_file_path: 真实的程序比例CSV文件路径
        train_h5ad_file_path: 训练数据文件路径(用于计算扰动均值)
        n_genes_for_metric: 用于计算转录组指标的基因数量
        random_seed: 随机种子(用于选择基因)
    """
    print("\n" + "=" * 50)
    print("STARTING EVALUATION")
    print("=" * 50)

    # 1. 加载数据
    print("Loading prediction data...")
    pred_adata = sc.read_h5ad(prediction_h5ad_file_path, backed='r')

    print("Loading ground truth data...")
    gtruth_adata = sc.read_h5ad(gtruth_h5ad_file_path, backed='r')

    print("Loading ground truth proportions...")
    true_proportions = pd.read_csv(gtruth_proportion_csv_file_path)

    print("Loading training data for perturbed centroid...")
    train_adata = sc.read_h5ad(train_h5ad_file_path, backed='r')

    # 2. 获取扰动列表
    perturbations = gtruth_adata.obs["gene"].cat.categories.tolist()
    print(f"Number of perturbations to evaluate: {len(perturbations)}")

    # 3. 选择用于计算的基因
    all_genes = pred_adata.var_names.tolist()
    n_available_genes = len(all_genes)

    if n_available_genes > n_genes_for_metric:
        print(f"Randomly selecting {n_genes_for_metric} genes out of {n_available_genes}")
        rng = np.random.default_rng(seed=random_seed)
        selected_genes = rng.choice(all_genes, size=n_genes_for_metric, replace=False).tolist()
    else:
        print(f"Using all {n_available_genes} available genes")
        selected_genes = all_genes

    print(f"Using {len(selected_genes)} genes for transcriptome-wide metrics")

    # 4. 计算训练集的扰动均值
    print("Computing perturbed centroid from training data...")
    control_label = "NC"
    perturbed_mask = train_adata.obs["gene"] != control_label

    # 分批计算扰动均值以节省内存
    batch_size = 5000
    perturbed_indices = np.where(perturbed_mask)[0]
    n_perturbed = len(perturbed_indices)

    # 只计算选中基因的均值
    gene_indices = [train_adata.var_names.tolist().index(gene) for gene in selected_genes]

    sum_expr = np.zeros(len(selected_genes))
    for i in range(0, n_perturbed, batch_size):
        end_idx = min(i + batch_size, n_perturbed)
        batch_indices = perturbed_indices[i:end_idx]

        # 读取批次数据
        batch_adata_rows = train_adata[batch_indices].to_memory()

        # 2. 在内存中的 AnnData 对象上，再按列名筛选基因
        if isinstance(batch_adata_rows.X, scipy_sparse.spmatrix):
            batch_data = batch_adata_rows[:, selected_genes].X.toarray()
        else:
            batch_data = batch_adata_rows[:, selected_genes].X

        sum_expr += batch_data.sum(axis=0)

    perturbed_centroid = sum_expr / n_perturbed
    print(f"Perturbed centroid computed from {n_perturbed} cells")

    # 5. 计算转录组水平指标
    print("\nComputing transcriptome-wide metrics...")
    pearson_scores = []
    mmd_scores = []

    for pert in tqdm(perturbations, desc="Evaluating perturbations"):
        # 获取真实和预测的细胞
        gt_mask = gtruth_adata.obs["gene"] == pert
        pred_mask = pred_adata.obs["gene"] == pert

        n_gt = gt_mask.sum()
        n_pred = pred_mask.sum()

        if n_gt == 0 or n_pred == 0:
            print(f"Warning: Missing data for {pert}, skipping")
            continue

        # 读取选中基因的表达数据
        gt_rows = gtruth_adata[gt_mask].to_memory()
        if isinstance(gt_rows.X, scipy_sparse.spmatrix):
            gt_X = gt_rows[:, selected_genes].X.toarray()
        else:
            gt_X = gt_rows[:, selected_genes].X

        # 读取预测数据
        pred_rows = pred_adata[pred_mask].to_memory()
        if isinstance(pred_rows.X, scipy_sparse.spmatrix):
            pred_X = pred_rows[:, selected_genes].X.toarray()
        else:
            pred_X = pred_rows[:, selected_genes].X

        # 计算Pearson Delta
        pearson = compute_pearson_delta(gt_X, pred_X, perturbed_centroid)
        pearson_scores.append(pearson)

        # 计算MMD (使用相同数量的样本)
        min_samples = min(n_gt, n_pred)
        gt_X_balanced = gt_X[:min_samples]
        pred_X_balanced = pred_X[:min_samples]
        mmd = compute_mmd(gt_X_balanced, pred_X_balanced)
        mmd_scores.append(mmd)

    # 6. 计算程序水平指标
    print("\nComputing program-level metrics...")
    pred_proportions = pd.read_csv(
        os.path.join(os.path.dirname(prediction_h5ad_file_path), "predict_program_proportion.csv"))
    l1_distance = compute_l1_distance(true_proportions, pred_proportions)

    # 7. 输出结果
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    if pearson_scores:
        print(f"Pearson Delta: {np.mean(pearson_scores):.4f} (higher is better)")
    if mmd_scores:
        print(f"MMD: {np.mean(mmd_scores):.4f} (lower is better)")
    print(f"L1 Distance: {l1_distance:.4f} (lower is better)")
    print("=" * 50 + "\n")

    return {
        "pearson": np.mean(pearson_scores) if pearson_scores else None,
        "mmd": np.mean(mmd_scores) if mmd_scores else None,
        "l1": l1_distance
    }
