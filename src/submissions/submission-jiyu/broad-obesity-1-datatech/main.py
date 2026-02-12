import os
import sys
import datetime
import pandas as pd
import scanpy as sc

# 导入拆分后的模块
from utils import Logger
from train import train
from infer import infer
from metrics import evaluate_predictions

# 设置随机种子
import numpy as np
np.random.seed(42)

# ==========================================
# Local Testing Block (Main Execution)
# ==========================================
if __name__ == "__main__":
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"execution_log_{current_time}.txt"
    sys.stdout = Logger(log_filename)

    print("=" * 40)
    print("STARTING LOCAL TEST EXECUTION")
    print("=" * 40)

    # 1. 定义路径
    # 使用提交目录中的数据路径
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    MODEL_DIR = os.path.join(BASE_DIR, "resources")
    PREDICTION_DIR = os.path.join(BASE_DIR, "prediction")

    # 确保目录存在
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PREDICTION_DIR, exist_ok=True)

    # 2. 模拟竞赛环境的数据准备
    # 注意：你需要确保 'obesity_challenge_1.h5ad' 和 'obesity_challenge_1_local_gtruth.h5ad' 在 data 目录下
    train_file = os.path.join(DATA_DIR, "obesity_challenge_1.h5ad")
    local_gtruth_file = os.path.join(DATA_DIR, "obesity_challenge_1_local_gtruth.h5ad")

    if not os.path.exists(train_file):
        print(f"Error: Training file not found at {train_file}")
        exit(1)

    # ==========================
    # Phase 1: Training
    # ==========================
    #print("\n>>> Phase 1: Running train()...")
    # 如果是为了快速测试代码跑通，可以暂时注释掉 train，前提是你已经有了保存好的模型
    train(data_directory_path=DATA_DIR, model_directory_path=MODEL_DIR)

    # ==========================
    # Phase 2: Inference Preparation
    # ==========================
    print("\n>>> Phase 2: Preparing for inference...")

    # A. 获取 genes_to_predict (列名)
    genes_to_predict_path = os.path.join(DATA_DIR, "genes_to_predict.txt")
    if os.path.exists(genes_to_predict_path):
        genes_to_predict = pd.read_csv(genes_to_predict_path, header=None)[0].values.tolist()
    else:
        print("Warning: genes_to_predict.txt not found. Using var_names from train data (slow).")
        temp_adata = sc.read_h5ad(train_file, backed='r')
        genes_to_predict = temp_adata.var_names.tolist()
        del temp_adata

    # B. 获取 predict_perturbations (本地测试使用 local_gtruth 中的基因)
    if os.path.exists(local_gtruth_file):
        print(f"Loading local ground truth from {local_gtruth_file}")
        gtruth = sc.read_h5ad(local_gtruth_file, backed='r')
        # 获取 local gtruth 中的 perturbation 列表
        predict_perturbations = gtruth.obs["gene"].cat.categories.tolist()
        print(f"Perturbations to predict (from local gtruth): {predict_perturbations}")
    else:
        print("Warning: Local ground truth not found. Defining dummy perturbations.")
        predict_perturbations = ["CHD4", "FOXC1"]  # 仅作为测试示例

    # ==========================
    # Phase 3: Inference
    # ==========================
    print("\n>>> Phase 3: Running infer()...")

    infer(
        data_directory_path=DATA_DIR,
        prediction_directory_path=PREDICTION_DIR,
        prediction_h5ad_file_path=os.path.join(PREDICTION_DIR, "prediction.h5ad"),
        program_proportion_csv_file_path=os.path.join(PREDICTION_DIR, "predict_program_proportion.csv"),
        model_directory_path=MODEL_DIR,
        predict_perturbations=predict_perturbations,
        genes_to_predict=genes_to_predict,
        cells_per_perturbation=100  # 生成 100 个细胞用于分布评估
    )

    print("\n" + "=" * 40)
    print("LOCAL TEST COMPLETED SUCCESSFULLY")
    print(f"Check outputs in: {PREDICTION_DIR}")
    print("=" * 40)

    print("\n>>> Phase 4: Running evaluation...")

    # 只有在本地测试集存在时才运行评估
    if os.path.exists(local_gtruth_file):
        gtruth_proportion_file = os.path.join(DATA_DIR, "program_proportion_local_gtruth.csv")
        if os.path.exists(gtruth_proportion_file):
            evaluate_predictions(
                prediction_h5ad_file_path=os.path.join(PREDICTION_DIR, "prediction.h5ad"),
                gtruth_h5ad_file_path=local_gtruth_file,
                gtruth_proportion_csv_file_path=gtruth_proportion_file,
                train_h5ad_file_path=train_file,
                n_genes_for_metric=1000,  # 与公开榜一致的基因数量
                random_seed=42
            )
        else:
            print("Warning: Ground truth proportion file not found, skipping evaluation")
    else:
        print("Info: Local ground truth not found, skipping evaluation")

    print("\n" + "=" * 40)
    print("LOCAL TEST COMPLETED SUCCESSFULLY")
    print(f"Check outputs in: {PREDICTION_DIR}")
    print("=" * 40)

    # Optional: 简单的结果检查
    if os.path.exists(os.path.join(PREDICTION_DIR, "prediction.h5ad")):
        pred = sc.read_h5ad(os.path.join(PREDICTION_DIR, "prediction.h5ad"), backed='r')
        print("\nPrediction Output Summary:")
        print(pred)
        print(f"Obs 'gene' categories: {pred.obs['gene'].unique().tolist()[:5]}...")
