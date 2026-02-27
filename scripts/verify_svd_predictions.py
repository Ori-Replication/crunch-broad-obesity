#!/usr/bin/env python3
"""
验证脚本：确认 submission-final 中的 SVD 算法与 tmp/main.py 生成的预测向量完全一致。

验证内容：
1. gene_embeddings (gene_map) - SVD 基因嵌入
2. control_mean - 控制组均值
3. shift_model - Ridge 回归模型（系数与截距）
4. pred_delta - 每个扰动基因的预测 Delta 向量（SVD+Ridge 核心输出）

使用方法：
    python scripts/verify_svd_predictions.py [DATA_DIR]

    若不指定 DATA_DIR，默认使用：
    - src/submissions/submission-final/data
    - 或 tmp/data（若前者不存在）
"""

import os
import sys
import gc
import numpy as np
import pandas as pd
import scanpy as sc
import joblib
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tmp"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "submissions" / "submission-final"))

# 固定随机种子，确保可复现
np.random.seed(42)


def get_default_data_dir():
    """获取默认数据目录"""
    candidates = [
        PROJECT_ROOT / "src" / "submissions" / "submission-final" / "data",
        PROJECT_ROOT / "tmp" / "data",
        PROJECT_ROOT / "data",
    ]
    for d in candidates:
        h5ad = d / "obesity_challenge_1.h5ad"
        if h5ad.exists():
            return str(d)
    return None


def run_tmp_train(data_dir: str, model_dir: str):
    """运行 tmp/main.py 的 train 逻辑"""
    from importlib.util import spec_from_file_location, module_from_spec
    tmp_main = PROJECT_ROOT / "tmp" / "main.py"
    spec = spec_from_file_location("tmp_main", tmp_main)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    module.train(data_directory_path=data_dir, model_directory_path=model_dir)


def run_final_train(data_dir: str, model_dir: str):
    """运行 submission-final 的 train"""
    from train import train as final_train
    final_train(data_directory_path=data_dir, model_directory_path=model_dir)


def load_tmp_artifacts(model_dir: str):
    """加载 tmp 版本保存的模型"""
    control_mean = np.load(os.path.join(model_dir, "control_mean.npy"))
    gene_map = joblib.load(os.path.join(model_dir, "gene_embeddings.pkl"))
    shift_model = joblib.load(os.path.join(model_dir, "shift_model.pkl"))
    return {"control_mean": control_mean, "gene_map": gene_map, "shift_model": shift_model}


def load_final_artifacts(model_dir: str):
    """加载 submission-final 版本保存的模型"""
    control_mean = joblib.load(os.path.join(model_dir, "control_mean.pkl"))
    control_mean = np.ravel(np.asarray(control_mean))
    gene_map = joblib.load(os.path.join(model_dir, "gene_embeddings.pkl"))
    shift_model = joblib.load(os.path.join(model_dir, "shift_model.pkl"))
    return {"control_mean": control_mean, "gene_map": gene_map, "shift_model": shift_model}


def compare_arrays(name: str, a: np.ndarray, b: np.ndarray, rtol=1e-5, atol=1e-8) -> bool:
    """比较两个数组是否一致"""
    a = np.asarray(a).flatten()
    b = np.asarray(b).flatten()
    if a.shape != b.shape:
        print(f"  [FAIL] {name}: shape 不一致 {a.shape} vs {b.shape}")
        return False
    if np.allclose(a, b, rtol=rtol, atol=atol):
        max_diff = np.max(np.abs(a - b))
        print(f"  [PASS] {name}: 完全一致 (max_diff={max_diff:.2e})")
        return True
    else:
        diff = np.abs(a - b)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        print(f"  [FAIL] {name}: 不一致 (max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e})")
        return False


def compare_gene_maps(name: str, map_a: dict, map_b: dict) -> bool:
    """比较两个 gene_map 是否一致"""
    genes_a = set(map_a.keys())
    genes_b = set(map_b.keys())
    if genes_a != genes_b:
        only_a = genes_a - genes_b
        only_b = genes_b - genes_a
        print(f"  [FAIL] {name}: 基因集合不一致")
        if only_a:
            print(f"    仅 tmp 有: {len(only_a)} 个")
        if only_b:
            print(f"    仅 final 有: {len(only_b)} 个")
        return False

    all_ok = True
    for g in list(genes_a)[:5]:  # 抽样检查前5个
        if not np.allclose(map_a[g], map_b[g], rtol=1e-5, atol=1e-8):
            print(f"  [FAIL] {name}: 基因 {g} 的 embedding 不一致")
            all_ok = False
            break

    if all_ok:
        # 全面检查
        for g in genes_a:
            if not np.allclose(map_a[g], map_b[g], rtol=1e-5, atol=1e-8):
                print(f"  [FAIL] {name}: 基因 {g} 的 embedding 不一致")
                all_ok = False
                break

    if all_ok:
        print(f"  [PASS] {name}: 所有 {len(genes_a)} 个基因的 embedding 一致")
    return all_ok


def compare_ridge_models(name: str, m1, m2) -> bool:
    """比较两个 Ridge 模型"""
    ok = True
    if not np.allclose(m1.coef_, m2.coef_, rtol=1e-5, atol=1e-8):
        print(f"  [FAIL] {name}: Ridge coef_ 不一致")
        ok = False
    else:
        print(f"  [PASS] {name}: Ridge coef_ 一致")

    if not np.allclose(m1.intercept_, m2.intercept_, rtol=1e-5, atol=1e-8):
        print(f"  [FAIL] {name}: Ridge intercept_ 不一致")
        ok = False
    else:
        print(f"  [PASS] {name}: Ridge intercept_ 一致")
    return ok


def get_predict_perturbations(data_dir: str) -> list:
    """获取 predict_perturbations 列表"""
    pert_file = os.path.join(data_dir, "predict_perturbations.txt")
    if os.path.exists(pert_file):
        return pd.read_csv(pert_file, header=None)[0].tolist()
    # 从 local_gtruth 获取
    gtruth_file = os.path.join(data_dir, "obesity_challenge_1_local_gtruth.h5ad")
    if os.path.exists(gtruth_file):
        gtruth = sc.read_h5ad(gtruth_file, backed="r")
        return gtruth.obs["gene"].cat.categories.tolist()
    # 从训练数据获取（排除 NC）
    adata = sc.read_h5ad(os.path.join(data_dir, "obesity_challenge_1.h5ad"), backed="r")
    perts = [p for p in adata.obs["gene"].unique() if p != "NC"]

    return perts[:20]  # 取前20个作为测试


def get_genes_to_predict(data_dir: str) -> list:
    """获取 genes_to_predict 列表"""
    genes_file = os.path.join(data_dir, "genes_to_predict.txt")
    if os.path.exists(genes_file):
        return pd.read_csv(genes_file, header=None)[0].tolist()
    adata = sc.read_h5ad(os.path.join(data_dir, "obesity_challenge_1.h5ad"), backed="r")
    return adata.var_names.tolist()


def compare_pred_deltas(
    tmp_artifacts: dict,
    final_artifacts: dict,
    data_dir: str,
    predict_perturbations: list,
    genes_to_predict: list,
) -> bool:
    """比较两个版本对每个扰动的 pred_delta 是否一致"""
    gene_map_tmp = tmp_artifacts["gene_map"]
    shift_model_tmp = tmp_artifacts["shift_model"]
    gene_map_final = final_artifacts["gene_map"]
    shift_model_final = final_artifacts["shift_model"]

    adata = sc.read_h5ad(os.path.join(data_dir, "obesity_challenge_1.h5ad"), backed="r")
    all_var_names = adata.var_names.tolist()
    gene_idx_map = {g: i for i, g in enumerate(all_var_names)}
    target_indices = [gene_idx_map[g] for g in genes_to_predict if g in gene_idx_map]

    default_emb_tmp = np.mean(list(gene_map_tmp.values()), axis=0)
    default_emb_final = np.mean(list(gene_map_final.values()), axis=0)

    all_ok = True
    failed_perts = []
    for pert in predict_perturbations:
        emb_tmp = gene_map_tmp.get(pert, default_emb_tmp).reshape(1, -1)
        emb_final = gene_map_final.get(pert, default_emb_final).reshape(1, -1)

        pred_delta_full_tmp = shift_model_tmp.predict(emb_tmp).flatten()
        pred_delta_full_final = shift_model_final.predict(emb_final).flatten()

        pred_delta_sub_tmp = pred_delta_full_tmp[target_indices]
        pred_delta_sub_final = pred_delta_full_final[target_indices]

        if not np.allclose(pred_delta_sub_tmp, pred_delta_sub_final, rtol=1e-5, atol=1e-8):
            failed_perts.append(pert)
            all_ok = False

    if all_ok:
        print(f"  [PASS] pred_delta: 全部 {len(predict_perturbations)} 个扰动的预测向量一致")
    else:
        print(f"  [FAIL] pred_delta: {len(failed_perts)} 个扰动不一致: {failed_perts[:10]}{'...' if len(failed_perts) > 10 else ''}")
    return all_ok


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else None
    if data_dir is None:
        data_dir = get_default_data_dir()
    if data_dir is None or not os.path.exists(os.path.join(data_dir, "obesity_challenge_1.h5ad")):
        print("错误: 未找到数据文件 obesity_challenge_1.h5ad")
        print("用法: python scripts/verify_svd_predictions.py <DATA_DIR>")
        print("示例: python scripts/verify_svd_predictions.py src/submissions/submission-final/data")
        sys.exit(1)

    proportion_path = os.path.join(data_dir, "program_proportion.csv")
    if not os.path.exists(proportion_path):
        print("错误: tmp/main.py 需要 program_proportion.csv 才能完成训练。")
        print(f"  请确保 {proportion_path} 存在。")
        sys.exit(1)

    tmp_model_dir = str(PROJECT_ROOT / "tmp_resources_verify")
    final_model_dir = str(PROJECT_ROOT / "final_resources_verify")
    os.makedirs(tmp_model_dir, exist_ok=True)
    os.makedirs(final_model_dir, exist_ok=True)

    print("=" * 60)
    print("SVD 预测向量一致性验证")
    print("=" * 60)
    print(f"数据目录: {data_dir}")
    print(f"tmp 模型输出: {tmp_model_dir}")
    print(f"final 模型输出: {final_model_dir}")
    print()

    # 1. 运行 tmp 训练
    print(">>> 步骤 1: 运行 tmp/main.py 的 train()...")
    try:
        run_tmp_train(data_dir, tmp_model_dir)
        print("tmp train 完成")
    except Exception as e:
        print(f"tmp train 失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    print()

    # 2. 运行 submission-final 训练
    # 重要：重置随机种子，确保与 tmp 使用相同的 NC 采样
    np.random.seed(42)
    print(">>> 步骤 2: 运行 submission-final 的 train()...")
    try:
        run_final_train(data_dir, final_model_dir)
        print("submission-final train 完成")
    except Exception as e:
        print(f"submission-final train 失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    print()

    # 3. 加载并比较
    print(">>> 步骤 3: 加载模型并比较...")
    tmp_artifacts = load_tmp_artifacts(tmp_model_dir)
    final_artifacts = load_final_artifacts(final_model_dir)

    results = []

    print("\n--- 比较 control_mean ---")
    results.append(compare_arrays("control_mean", tmp_artifacts["control_mean"], final_artifacts["control_mean"]))

    print("\n--- 比较 gene_embeddings (gene_map) ---")
    results.append(compare_gene_maps("gene_map", tmp_artifacts["gene_map"], final_artifacts["gene_map"]))

    print("\n--- 比较 shift_model (Ridge) ---")
    results.append(compare_ridge_models("shift_model", tmp_artifacts["shift_model"], final_artifacts["shift_model"]))

    print("\n--- 比较 pred_delta (每个扰动的预测向量) ---")
    predict_perturbations = get_predict_perturbations(data_dir)
    genes_to_predict = get_genes_to_predict(data_dir)
    print(f"  测试 {len(predict_perturbations)} 个扰动, {len(genes_to_predict)} 个基因")
    results.append(
        compare_pred_deltas(
            tmp_artifacts, final_artifacts,
            data_dir, predict_perturbations, genes_to_predict
        )
    )

    # 4. 总结
    print("\n" + "=" * 60)
    if all(results):
        print("验证结果: 全部通过")
        print("submission-final 的 SVD 算法与 tmp 版本生成的预测向量完全一致。")
    else:
        print("验证结果: 存在不一致")
        print("请检查上述 [FAIL] 项。")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
