"""
预计算脚本：使用 PM+FlowImp_ode100_pc5_a0.11（Flow Matching 改进版）生成预测表
本脚本在本地运行，生成 resources/ 中的所有文件

方法：流匹配组合 (1-α)*PerturbMean + α*FlowMatch，α=0.11
- 纯 PCA 流匹配：在 delta 空间学习 p(扰动 PC scores | 基因 PC loadings)
- 改进：ODE 积分步数 100（原 50），α=0.11（原 0.10）
- 5 折 CV 最佳：Pearson 0.2301 vs PerturbMean 0.2215 (+3.9%)

运行方式：
    cd CrunchDAO-obesity
    conda activate broad
    python src/submissions/submission-flow/broad-obesity-1-datatech/prepare_resources.py
"""

import os
import gc
import numpy as np
import pandas as pd
import scanpy as sc
import anndata
import joblib
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from sklearn.decomposition import PCA

# 路径配置
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "original_data" / "default"
RESOURCES_DIR = SCRIPT_DIR / "resources"
GENE2VEC_SRC = PROJECT_ROOT / "external_repos" / "Gene2vec" / "pre_trained_emb" / "gene2vec_dim_200_iter_9_w2v.txt"

RESOURCES_DIR.mkdir(exist_ok=True, parents=True)


def to_array(x):
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "toarray"):
        return x.toarray()
    return np.array(x)


# =============================================================================
# Flow Matching Model (from eda/flow)
# =============================================================================


class ConditionalVelocityNet(nn.Module):
    """MLP 预测速度场 v(x, t, condition)。"""

    def __init__(self, dim_x: int, dim_cond: int, hidden: int = 128, n_layers: int = 4):
        super().__init__()
        layers = []
        in_dim = dim_x + 1 + dim_cond
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(in_dim, hidden), nn.SiLU()])
            in_dim = hidden
        layers.append(nn.Linear(in_dim, dim_x))
        self.net = nn.Sequential(*layers)

    def forward(self, x, t, cond):
        t_vec = t.view(-1, 1).expand(x.shape[0], 1)
        inp = torch.cat([x, t_vec, cond], dim=-1)
        return self.net(inp)


class FlowMatchingPCA:
    """纯 PCA 流匹配预测器。"""

    def __init__(
        self,
        n_components: int = 10,
        hidden: int = 64,
        n_layers: int = 3,
        lr: float = 5e-4,
        epochs: int = 800,
        batch_size: int = 32,
        device: str = "cpu",
        n_samples: int = 10,
        deterministic: bool = False,
        n_ode_steps: int = 100,
    ):
        self.n_components = n_components
        self.n_ode_steps = n_ode_steps
        self.hidden = hidden
        self.n_layers = n_layers
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.n_samples = n_samples
        self.deterministic = deterministic
        self.fallback = None
        self.ok = False

    def fit(self, td, tp, gene_names=None, **kw):
        self.gn = gene_names or [f"g{i}" for i in range(td.shape[1])]
        Y = td.T  # (n_genes, n_train_perts)

        n_pc = min(self.n_components, Y.shape[0] - 1, Y.shape[1] - 1)
        if n_pc < 2:
            self.fallback = td.mean(axis=0)
            return

        self.pca_genes = PCA(n_components=n_pc)
        gene_loadings = self.pca_genes.fit_transform(Y)

        self.pca_perts = PCA(n_components=n_pc)
        pert_scores = self.pca_perts.fit_transform(td)

        self.gene_loadings_df = pd.DataFrame(gene_loadings, index=self.gn)
        train_perts = [p for p in tp if p in self.gene_loadings_df.index]
        if len(train_perts) < n_pc + 2:
            self.fallback = td.mean(axis=0)
            return

        cond_list, target_list = [], []
        for p in train_perts:
            c = self.gene_loadings_df.loc[p].values.astype(np.float32)
            idx = tp.index(p)
            y = pert_scores[idx].astype(np.float32)
            cond_list.append(c)
            target_list.append(y)

        cond_arr = np.array(cond_list)
        target_arr = np.array(target_list)

        self.cond_mean = cond_arr.mean(axis=0)
        self.cond_std = cond_arr.std(axis=0) + 1e-6
        cond_norm = (cond_arr - self.cond_mean) / self.cond_std

        self.target_mean = target_arr.mean(axis=0)
        self.target_std = target_arr.std(axis=0) + 1e-6
        target_norm = (target_arr - self.target_mean) / self.target_std

        self.model = ConditionalVelocityNet(
            dim_x=n_pc, dim_cond=n_pc, hidden=self.hidden, n_layers=self.n_layers
        ).to(self.device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)

        n_train = len(cond_list)
        cond_t = torch.from_numpy(cond_norm).float().to(self.device)
        target_t = torch.from_numpy(target_norm).float().to(self.device)

        self.model.train()
        for ep in range(self.epochs):
            perm = np.random.permutation(n_train)
            for i in range(0, n_train, self.batch_size):
                idx = perm[i : i + self.batch_size]
                if len(idx) < 2:
                    continue
                c = cond_t[idx]
                x1 = target_t[idx]
                x0 = torch.randn_like(x1, device=self.device)
                t = torch.rand(len(idx), device=self.device)
                xt = (1 - t).unsqueeze(-1) * x0 + t.unsqueeze(-1) * x1
                v_target = x1 - x0
                v_pred = self.model(xt, t, c)
                loss = ((v_pred - v_target) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()

        self.fallback = td.mean(axis=0)
        self.ok = True

    def predict_single(self, gene_name):
        """预测单个基因扰动的 delta。若基因不在训练载荷中则返回 fallback。"""
        if not self.ok or gene_name not in self.gene_loadings_df.index:
            return self.fallback.copy()

        n_genes = len(self.gn)
        self.model.eval()
        with torch.no_grad():
            c_raw = self.gene_loadings_df.loc[gene_name].values.astype(np.float32)
            c_norm = (c_raw - self.cond_mean) / self.cond_std
            c = torch.from_numpy(c_norm).float().to(self.device)

            preds = []
            for _ in range(self.n_samples if not self.deterministic else 1):
                x = (
                    torch.zeros(1, self.n_components, device=self.device)
                    if self.deterministic
                    else torch.randn(1, self.n_components, device=self.device)
                )
                c_batch = c.unsqueeze(0).expand(1, -1)
                for k in range(self.n_ode_steps):
                    t = torch.tensor([k / self.n_ode_steps], device=self.device)
                    v = self.model(x, t, c_batch)
                    x = x + v / self.n_ode_steps
                x_np = x.cpu().numpy().flatten()
                scores = x_np * self.target_std + self.target_mean
                delta_pc = self.pca_perts.inverse_transform(
                    scores.reshape(1, -1)
                ).flatten()
                preds.append(delta_pc[:n_genes])
            return np.mean(preds, axis=0)


def main():
    print("=" * 80)
    print("Prepare Resources: PM+FlowImp_ode100_pc5_a0.11 (Flow Matching) Method")
    print("=" * 80, flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    # ============================================
    # 1. 加载竞赛数据
    # ============================================
    print("\n[1] Loading competition data...", flush=True)
    adata = sc.read_h5ad(DATA_DIR / "obesity_challenge_1.h5ad", backed="r")
    gene_names = adata.var.index.tolist()
    n_genes = len(gene_names)

    nc_indices = np.where(adata.obs["gene"] == "NC")[0]
    X_nc = to_array(adata[nc_indices].X)
    control_mean = X_nc.mean(axis=0)
    print(f"  NC cells: {len(nc_indices)}, genes: {n_genes}", flush=True)

    unique_perts = [p for p in adata.obs["gene"].cat.categories if p != "NC"]
    print(f"  Training perturbations: {len(unique_perts)}", flush=True)

    train_deltas = {}
    for pert in tqdm(unique_perts, desc="  Computing train deltas"):
        mask = adata.obs["gene"] == pert
        if mask.sum() > 0:
            X_pert = to_array(adata[mask].X)
            pert_mean = X_pert.mean(axis=0)
            train_deltas[pert] = pert_mean - control_mean

    train_perts = list(train_deltas.keys())
    train_deltas_np = np.array([train_deltas[p] for p in train_perts])
    print(f"  Valid training perturbations: {len(train_perts)}", flush=True)

    average_delta = train_deltas_np.mean(axis=0)

    predict_perts_file = DATA_DIR / "predict_perturbations.txt"
    predict_perturbations = pd.read_csv(predict_perts_file, header=None)[0].tolist()
    print(f"  Prediction targets: {len(predict_perturbations)}", flush=True)

    proportion_path = DATA_DIR / "program_proportion.csv"
    df_prop = pd.read_csv(proportion_path)
    avg_prop = df_prop[["pre_adipo", "adipo", "lipo", "other"]].mean(axis=0).values
    if avg_prop[2] > avg_prop[1]:
        avg_prop[2] = avg_prop[1]
    sm = avg_prop[0] + avg_prop[1] + avg_prop[3]
    if sm > 0:
        avg_prop[0] /= sm
        avg_prop[1] /= sm
        avg_prop[3] /= sm
    print(
        f"  Avg proportions: pre={avg_prop[0]:.4f} adipo={avg_prop[1]:.4f} "
        f"lipo={avg_prop[2]:.4f} other={avg_prop[3]:.4f}",
        flush=True,
    )

    # ============================================
    # 2. 训练 PM+FlowImp_ode100_pc5_a0.11（改进版：ODE 100 步，α=0.11）
    # ============================================
    print("\n[2] Training PM+FlowImp_ode100_pc5_a0.11 (PerturbMean + FlowMatch, α=0.11, n_ode=100)...", flush=True)

    pm_delta = average_delta.copy()
    flow = FlowMatchingPCA(
        n_components=5,
        hidden=64,
        n_layers=3,
        epochs=800,
        batch_size=min(32, len(train_perts) - 1),
        device=device,
        n_samples=10,
        n_ode_steps=100,
    )
    flow.fit(train_deltas_np, train_perts, gene_names=gene_names)
    alpha = 0.11
    print(f"  Flow matching trained. Using α={alpha}, n_ode_steps=100", flush=True)

    # ============================================
    # 3. 生成预测表
    # ============================================
    print("\n[3] Generating prediction table...", flush=True)

    prediction_means = {}
    n_flow_pred = 0
    n_fallback = 0

    for pert in tqdm(predict_perturbations, desc="  Predicting"):
        flow_delta = flow.predict_single(pert)
        delta = (1 - alpha) * pm_delta + alpha * flow_delta
        prediction_means[pert] = (control_mean + delta).astype(np.float32)
        if flow.ok and pert in flow.gene_loadings_df.index:
            n_flow_pred += 1
        else:
            n_fallback += 1

    print(f"  Flow predictions (gene in loadings): {n_flow_pred}", flush=True)
    print(f"  Fallback (PerturbMean only): {n_fallback}", flush=True)

    pred_matrix = np.array([prediction_means[p] for p in predict_perturbations])
    pred_obs = pd.DataFrame({"gene": predict_perturbations})
    pred_var = pd.DataFrame(index=gene_names)
    pred_adata = anndata.AnnData(X=pred_matrix, obs=pred_obs, var=pred_var)

    pred_adata.X = pred_adata.X.astype(np.float16)
    pred_path = RESOURCES_DIR / "prediction_table_fp16_compressed.h5ad"
    pred_adata.write_h5ad(pred_path)
    print(f"  Saved: {pred_path} ({os.path.getsize(pred_path) / 1e6:.1f} MB)", flush=True)

    # ============================================
    # 4. 保存其他资源
    # ============================================
    print("\n[4] Saving resources...", flush=True)

    # 复制 Gene2vec 到 resources，供 ProportionKNNPredictor 使用
    if GENE2VEC_SRC.exists():
        import shutil

        gene2vec_dst = RESOURCES_DIR / "gene2vec_dim_200_iter_9_w2v.txt"
        shutil.copy2(GENE2VEC_SRC, gene2vec_dst)
        print(f"  Copied Gene2vec to: {gene2vec_dst}", flush=True)
    else:
        print(f"  Warning: Gene2vec not found at {GENE2VEC_SRC}", flush=True)

    joblib.dump(average_delta, RESOURCES_DIR / "average_delta.pkl")
    joblib.dump(avg_prop, RESOURCES_DIR / "average_proportions.pkl")
    joblib.dump(control_mean, RESOURCES_DIR / "control_mean.pkl")
    joblib.dump(nc_indices, RESOURCES_DIR / "nc_indices.pkl")

    print(
        "  Saved: average_delta.pkl, average_proportions.pkl, control_mean.pkl, nc_indices.pkl",
        flush=True,
    )

    # ============================================
    # 5. 验证
    # ============================================
    print("\n[5] Validation...", flush=True)
    pred_check = sc.read_h5ad(pred_path)
    print(f"  Prediction table shape: {pred_check.shape}", flush=True)
    print(f"  Perturbations: {len(pred_check.obs['gene'].unique())}", flush=True)
    print(f"  X dtype: {pred_check.X.dtype}", flush=True)
    print(
        f"  X range: [{pred_check.X.min():.4f}, {pred_check.X.max():.4f}]",
        flush=True,
    )

    print("\n" + "=" * 80)
    print("Resources prepared successfully!")
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
