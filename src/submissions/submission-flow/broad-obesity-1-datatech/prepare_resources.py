"""
预计算脚本：使用 FlowXA_pc10_xpc20（纯 Flow Matching + xatlas）
本脚本在本地运行，生成 resources/ 中的所有文件

方法：纯 Flow Matching with xatlas（不组合 PerturbMean）
- n_components=10: PCA components for competition data
- n_xatlas_pc=20: PCA components for xatlas HEK293T
- 条件: [gene_loadings | xatlas_PCA_features]
- CV 结果: Pearson=0.1767, CosineSim=0.5442 (与 PM 差异大，不过拟合)

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
XATLAS_DIR = PROJECT_ROOT / "data" / "external_data" / "xatlas" / "processed"
GSE217812_DIR = PROJECT_ROOT / "data" / "external_data" / "GSE217812" / "processed"
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
# Flow Matching Model with xatlas (from eda/flow_xatlas)
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


class FlowMatchingPCA_xatlas:
    """
    Flow matching conditioned on [gene_loadings | xatlas_PCA_features].
    纯 Flow，不组合 PerturbMean。

    - gene_loadings: from competition PCA (genes x train_perts)
    - xatlas_features: PCA on xatlas HEK293T deltas
    - For genes not in xatlas: xatlas part = zeros
    """

    def __init__(
        self,
        n_components: int = 10,
        n_xatlas_pc: int = 20,
        hidden: int = 64,
        n_layers: int = 3,
        lr: float = 5e-4,
        epochs: int = 800,
        batch_size: int = 32,
        device: str = "cpu",
        n_samples: int = 10,
    ):
        self.n_components = n_components
        self.n_xatlas_pc = n_xatlas_pc
        self.hidden = hidden
        self.n_layers = n_layers
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.n_samples = n_samples
        self.xe = None
        self.xidx = None
        self.fallback = None
        self.ok = False

    def fit(self, td, tp, gene_names=None, xatlas_df=None, **kw):
        self.gn = gene_names or [f"g{i}" for i in range(td.shape[1])]
        Y = td.T  # (n_genes, n_train_perts)

        n_pc = min(self.n_components, Y.shape[0] - 1, Y.shape[1] - 1)
        if n_pc < 2:
            self.fallback = td.mean(axis=0)
            return

        # PCA on genes -> gene loadings
        self.pca_genes = PCA(n_components=n_pc)
        gene_loadings = self.pca_genes.fit_transform(Y)
        self.gene_loadings_df = pd.DataFrame(gene_loadings, index=self.gn)

        # PCA on perturbations -> target
        self.pca_perts = PCA(n_components=n_pc)
        pert_scores = self.pca_perts.fit_transform(td)

        # xatlas: PCA on deltas for perturbations in xatlas
        self.xe = xatlas_df
        dim_cond = n_pc
        self.xatlas_pca = None
        self.xatlas_mean = None
        self.xatlas_std = None
        self.xatlas_idx = {}

        if xatlas_df is not None and len(xatlas_df) > 0:
            xatlas_perts = [p for p in tp if p in xatlas_df.index]
            if len(xatlas_perts) >= min(5, self.n_xatlas_pc + 2):
                xf = xatlas_df.loc[xatlas_perts].values.astype(np.float32)
                nxpc = min(self.n_xatlas_pc, len(xatlas_perts) - 1, xf.shape[1] - 1)
                if nxpc >= 1:
                    self.xatlas_pca = PCA(n_components=nxpc)
                    xatlas_scores = self.xatlas_pca.fit_transform(xf)
                    self.xatlas_mean = xatlas_scores.mean(axis=0)
                    self.xatlas_std = xatlas_scores.std(axis=0) + 1e-6
                    self.xatlas_idx = {p: i for i, p in enumerate(xatlas_perts)}
                    dim_cond = n_pc + nxpc

        # Build condition: [gene_loadings | xatlas_features]
        train_perts = [p for p in tp if p in self.gene_loadings_df.index]
        if len(train_perts) < n_pc + 2:
            self.fallback = td.mean(axis=0)
            return

        cond_list, target_list = [], []
        for p in train_perts:
            c_gene = self.gene_loadings_df.loc[p].values.astype(np.float32)
            if self.xatlas_idx is not None and p in self.xatlas_idx:
                xatlas_raw = self.xatlas_pca.transform(
                    xatlas_df.loc[p].values.reshape(1, -1)
                )[0]
                xatlas_feat = ((xatlas_raw - self.xatlas_mean) / self.xatlas_std).astype(
                    np.float32
                )
                c = np.concatenate([c_gene, xatlas_feat])
            else:
                zeros = np.zeros(dim_cond - n_pc, dtype=np.float32)
                c = np.concatenate([c_gene, zeros])
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
            dim_x=n_pc,
            dim_cond=dim_cond,
            hidden=self.hidden,
            n_layers=self.n_layers,
        ).to(self.device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        cond_t = torch.from_numpy(cond_norm).float().to(self.device)
        target_t = torch.from_numpy(target_norm).float().to(self.device)
        n_train = len(cond_list)

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
        self.dim_cond = dim_cond
        self.n_xatlas_pc_act = dim_cond - n_pc if self.xatlas_pca is not None else 0
        self.ok = True

    def _get_cond(self, p, xatlas_df=None):
        """Get condition for perturbation p."""
        if p not in self.gene_loadings_df.index:
            return None
        c_gene = self.gene_loadings_df.loc[p].values.astype(np.float32)
        xe = xatlas_df if xatlas_df is not None else self.xe
        if self.xatlas_pca is not None and xe is not None and p in xe.index:
            xatlas_feat = (
                (
                    self.xatlas_pca.transform(
                        xe.loc[p].values.reshape(1, -1)
                    )[0]
                    - self.xatlas_mean
                )
                / self.xatlas_std
            )
            c = np.concatenate([c_gene, xatlas_feat.astype(np.float32)])
        else:
            zeros = np.zeros(self.dim_cond - len(c_gene), dtype=np.float32)
            c = np.concatenate([c_gene, zeros])
        return (c - self.cond_mean) / self.cond_std

    def predict_single(self, gene_name, xatlas_df=None):
        """预测单个基因扰动的 delta。若基因不在训练载荷中则返回 fallback。"""
        if not self.ok or gene_name not in self.gene_loadings_df.index:
            return self.fallback.copy()

        n_genes = len(self.gn)
        self.model.eval()
        with torch.no_grad():
            c = self._get_cond(gene_name, xatlas_df)
            if c is None:
                return self.fallback.copy()

            c_t = torch.from_numpy(c).float().to(self.device)
            preds = []
            for _ in range(self.n_samples):
                x = torch.randn(1, self.n_components, device=self.device)
                c_batch = c_t.unsqueeze(0).expand(1, -1)
                for k in range(50):
                    t = torch.tensor([k / 50], device=self.device)
                    v = self.model(x, t, c_batch)
                    x = x + v / 50
                x_np = x.cpu().numpy().flatten()
                scores = x_np * self.target_std + self.target_mean
                delta_pc = self.pca_perts.inverse_transform(
                    scores.reshape(1, -1)
                ).flatten()
                preds.append(delta_pc[:n_genes])
            return np.mean(preds, axis=0)


def agg_xatlas(perts, comp_genes, cell_line="HEK293T"):
    """Aggregate xatlas deltas."""
    bd = XATLAS_DIR / "by_batch"
    bf = sorted(
        [
            f
            for f in os.listdir(bd)
            if f.startswith(cell_line + "_") and f.endswith("_aggregated.parquet")
        ]
    )
    if not bf:
        return None
    fb = pd.read_parquet(bd / bf[0])
    og = sorted(list(set(comp_genes) & set(fb.columns)))
    del fb
    gc.collect()
    ts = set(perts) | {"Non-Targeting"}
    ds, dc = {}, {}
    for fn in tqdm(bf, desc=f"  Agg {cell_line}"):
        try:
            b = pd.read_parquet(bd / fn, columns=og)
            if "Non-Targeting" not in b.index:
                del b
                gc.collect()
                continue
            ctrl = b.loc["Non-Targeting"].values
            for p in set(b.index) & ts - {"Non-Targeting"}:
                d = b.loc[p].values - ctrl
                if p not in ds:
                    ds[p] = np.zeros(len(og), dtype=np.float64)
                    dc[p] = 0
                ds[p] += d
                dc[p] += 1
            del b
            gc.collect()
        except Exception:
            continue
    avg = {p: ds[p] / dc[p] for p in ds if dc[p] > 0}
    xd = pd.DataFrame(avg, index=og).T
    return xd


def main():
    print("=" * 80)
    print("Prepare Resources: FlowXA_pc10_xpc20 (Pure Flow Matching + xatlas)")
    print("=" * 80, flush=True)

    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

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

    # 可选：GSE217812 脂肪细胞池化 delta 增强 PCA 空间
    use_gse217812 = False
    gse217812_path = GSE217812_DIR / "gse217812_pooled_delta.parquet"
    if gse217812_path.exists():
        print("\n[1b] Loading GSE217812 adipocyte pooled delta...", flush=True)
        gse_df = pd.read_parquet(gse217812_path)
        # 对齐到竞赛基因
        overlap = [g for g in gene_names if g in gse_df.columns]
        if len(overlap) >= 1000:
            pooled_delta = np.zeros(n_genes, dtype=np.float64)
            for i, g in enumerate(gene_names):
                if g in gse_df.columns:
                    pooled_delta[i] = gse_df.loc["pooled_adipocyte", g]
            # 归一化到与 train_deltas 相近的尺度，作为额外行参与 PCA
            scale = np.std(train_deltas_np) / (np.std(pooled_delta) + 1e-8)
            pooled_delta = pooled_delta * min(scale, 2.0)
            train_deltas_np = np.vstack([train_deltas_np, pooled_delta.reshape(1, -1)])
            train_perts = train_perts + ["pooled_adipocyte"]  # 保持 td/tp 长度一致，Flow 训练时会过滤
            use_gse217812 = True
            print(f"  GSE217812: {len(overlap)} genes, augmented PCA with adipocyte prior", flush=True)
        else:
            print(f"  GSE217812: too few overlapping genes ({len(overlap)}), skipping", flush=True)
    else:
        print(f"\n  (GSE217812 not found at {gse217812_path}, run eda/GSE217812/process_gse217812.py first)", flush=True)

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
    # 2. 加载 xatlas HEK293T 数据
    # ============================================
    print("\n[2] Loading xatlas HEK293T data...", flush=True)
    xatlas_df = agg_xatlas(predict_perturbations, gene_names, cell_line="HEK293T")
    if xatlas_df is None:
        raise RuntimeError("Failed to load xatlas HEK293T data")
    print(f"  xatlas perturbations: {len(xatlas_df)}", flush=True)

    # ============================================
    # 3. 训练 FlowXA_pc10_xpc20（纯 Flow，不组合 PM）
    # ============================================
    print("\n[3] Training FlowXA_pc10_xpc20 (Pure Flow Matching + xatlas)...", flush=True)

    flow = FlowMatchingPCA_xatlas(
        n_components=10,
        n_xatlas_pc=20,
        hidden=64,
        n_layers=3,
        epochs=800,
        batch_size=min(32, len(train_perts) - 1),
        device=device,
        n_samples=10,
    )
    flow.fit(train_deltas_np, train_perts, gene_names=gene_names, xatlas_df=xatlas_df)
    print(f"  Flow matching trained. Using pure Flow (no PerturbMean)", flush=True)

    # ============================================
    # 4. 生成预测表
    # ============================================
    print("\n[4] Generating prediction table...", flush=True)

    prediction_means = {}
    n_flow_pred = 0
    n_fallback = 0

    for pert in tqdm(predict_perturbations, desc="  Predicting"):
        delta = flow.predict_single(pert, xatlas_df=xatlas_df)
        prediction_means[pert] = (control_mean + delta).astype(np.float32)
        if flow.ok and pert in flow.gene_loadings_df.index:
            n_flow_pred += 1
        else:
            n_fallback += 1

    print(f"  Flow predictions (gene in loadings): {n_flow_pred}", flush=True)
    print(f"  Fallback: {n_fallback}", flush=True)

    pred_matrix = np.array([prediction_means[p] for p in predict_perturbations])
    pred_obs = pd.DataFrame({"gene": predict_perturbations})
    pred_var = pd.DataFrame(index=gene_names)
    pred_adata = anndata.AnnData(X=pred_matrix, obs=pred_obs, var=pred_var)

    pred_adata.X = pred_adata.X.astype(np.float16)
    pred_path = RESOURCES_DIR / "prediction_table_fp16_compressed.h5ad"
    pred_adata.write_h5ad(pred_path)
    print(f"  Saved: {pred_path} ({os.path.getsize(pred_path) / 1e6:.1f} MB)", flush=True)

    # ============================================
    # 5. 保存其他资源
    # ============================================
    print("\n[5] Saving resources...", flush=True)

    # 提取 scGPT 嵌入供 ProportionKNNPredictor 使用（实验结论：scGPT+PCA32+K15 最优）
    all_pert_genes = list(dict.fromkeys(predict_perturbations + train_perts))
    SCGPT_DIR = PROJECT_ROOT / "data" / "external_model" / "scgpt"
    scgpt_model_path = SCGPT_DIR / "model.safetensors"
    scgpt_vocab_path = SCGPT_DIR / "vocab.json"
    if scgpt_model_path.exists() and scgpt_vocab_path.exists():
        try:
            import json
            from safetensors import safe_open

            with open(scgpt_vocab_path) as f:
                vocab = json.load(f)
            gene_vocab = {k: v for k, v in vocab.items() if k.strip() != ""}
            id_to_gene = {int(v): k for k, v in gene_vocab.items()}

            emb_key = None
            with safe_open(scgpt_model_path, framework="pt", device="cpu") as f:
                keys = list(f.keys())
            for k in ["encoder.weight", "gene_encoder.embedding.weight", "transformer.wte.weight"]:
                if k in keys:
                    emb_key = k
                    break
            if emb_key is None:
                emb_key = [k for k in keys if "embed" in k.lower() or "encoder" in k.lower()]
                emb_key = emb_key[0] if emb_key else keys[0]

            with safe_open(scgpt_model_path, framework="pt", device="cpu") as f:
                emb_np = np.asarray(f.get_tensor(emb_key))
            full_emb = {gene: emb_np[gid].astype(np.float32) for gid, gene in id_to_gene.items() if gid < emb_np.shape[0]}
            scgpt_subset = {g: full_emb[g] for g in all_pert_genes if g in full_emb}
            if len(scgpt_subset) >= 2:
                joblib.dump(scgpt_subset, RESOURCES_DIR / "scgpt_embeddings.pkl")
                print(f"  Extracted scGPT embeddings for {len(scgpt_subset)} genes -> scgpt_embeddings.pkl", flush=True)
            else:
                print(f"  Warning: Too few genes in scGPT vocab ({len(scgpt_subset)}), skipping scgpt_embeddings.pkl", flush=True)
        except Exception as e:
            print(f"  Warning: Failed to extract scGPT embeddings: {e}", flush=True)
    else:
        print(f"  Info: scGPT model not found, skipping scgpt_embeddings.pkl", flush=True)

    # 提取 Gene2vec 子集（仅所需基因）为紧凑 pkl，避免加载大文件
    all_pert_set = set(all_pert_genes)
    if GENE2VEC_SRC.exists():
        try:
            g2v = {}
            with open(GENE2VEC_SRC) as f:
                first = f.readline().strip().split()
                dim = int(first[1])
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < dim + 1:
                        continue
                    gene = parts[0]
                    if gene in all_pert_set:
                        g2v[gene] = np.array([float(x) for x in parts[1 : dim + 1]], dtype=np.float32)
            if len(g2v) >= 2:
                joblib.dump(g2v, RESOURCES_DIR / "gene2vec_embeddings.pkl")
                print(f"  Extracted Gene2vec embeddings for {len(g2v)} genes -> gene2vec_embeddings.pkl", flush=True)
            else:
                print(f"  Warning: Too few genes in Gene2vec ({len(g2v)}), skipping gene2vec_embeddings.pkl", flush=True)
        except Exception as e:
            print(f"  Warning: Failed to extract Gene2vec: {e}", flush=True)
    else:
        print(f"  Warning: Gene2vec not found at {GENE2VEC_SRC}", flush=True)

    average_delta = train_deltas_np.mean(axis=0)
    joblib.dump(average_delta, RESOURCES_DIR / "average_delta.pkl")
    joblib.dump(avg_prop, RESOURCES_DIR / "average_proportions.pkl")
    joblib.dump(control_mean, RESOURCES_DIR / "control_mean.pkl")
    joblib.dump(nc_indices, RESOURCES_DIR / "nc_indices.pkl")

    print(
        "  Saved: average_delta.pkl, average_proportions.pkl, control_mean.pkl, nc_indices.pkl",
        flush=True,
    )

    # ============================================
    # 6. 验证
    # ============================================
    print("\n[6] Validation...", flush=True)
    pred_check = sc.read_h5ad(pred_path)
    print(f"  Prediction table shape: {pred_check.shape}", flush=True)
    print(f"  Perturbations: {len(pred_check.obs['gene'].unique())}", flush=True)
    print(f"  X dtype: {pred_check.X.dtype}", flush=True)
    print(
        f"  X range: [{pred_check.X.min():.4f}, {pred_check.X.max():.4f}]",
        flush=True,
    )

    method_name = "FlowXA_pc10_xpc20+GSE217812" if use_gse217812 else "FlowXA_pc10_xpc20"
    print("\n" + "=" * 80)
    print("Resources prepared successfully!")
    print(f"Method: {method_name} (Pure Flow + xatlas)")
    print("CV: Pearson=0.1767, CosineSim=0.5442")
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
