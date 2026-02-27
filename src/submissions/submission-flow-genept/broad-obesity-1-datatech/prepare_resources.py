"""
预计算脚本：FlowXA_pc10_xpc20 + GenePT (Pearson 预测)

本脚本在本地运行，生成 resources/ 中的所有文件。
支持 --steps 参数，仅运行需要的环节。

步骤：
  all (默认): 全部
  pearson: 仅 Pearson 相关（Flow+GenePT 训练 + 预测表 + 基础 pkl），跳过 proportion 用 embedding
  flow: 仅 Flow+GenePT 训练 + 预测表 (1-4)，跳过 embedding 提取
  embeddings: 仅提取 embedding (genept/scgpt/gene2vec)，需 prediction_table 等已存在
  genept: 仅提取 GenePT embedding

运行方式：
    cd CrunchDAO-obesity
    conda activate broad
    python .../prepare_resources.py --steps pearson   # 仅 Pearson 预测相关
    python .../prepare_resources.py --steps genept    # 仅 GenePT
"""

import argparse
import os
import gc
import pickle
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
GENEPT_ADA_PATH = PROJECT_ROOT / "data" / "external_data" / "GenePT" / "GenePT_emebdding_v2" / "GenePT_gene_embedding_ada_text.pickle"
FLOW_RESOURCES = SCRIPT_DIR.parent.parent / "submission-flow" / "broad-obesity-1-datatech" / "resources"

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


class FlowMatchingPCA_xatlas_genept:
    """
    Flow matching conditioned on [gene_loadings | xatlas_PCA | genept_PCA].
    纯 Flow，不组合 PerturbMean。GenePT 条件扩展提升 Pearson。
    """

    def __init__(
        self,
        n_components: int = 10,
        n_xatlas_pc: int = 20,
        n_genept_pc: int = 20,
        hidden: int = 64,
        n_layers: int = 3,
        lr: float = 5e-4,
        epochs: int = 800,
        batch_size: int = 32,
        device: str = "cpu",
        n_samples: int = 10,
        use_genept: bool = True,
    ):
        self.n_components = n_components
        self.n_xatlas_pc = n_xatlas_pc
        self.n_genept_pc = n_genept_pc
        self.use_genept = use_genept
        self.hidden = hidden
        self.n_layers = n_layers
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.n_samples = n_samples
        self.xe = None
        self.genept_df = None
        self.fallback = None
        self.ok = False

    def fit(self, td, tp, gene_names=None, xatlas_df=None, genept_df=None, **kw):
        self.gn = gene_names or [f"g{i}" for i in range(td.shape[1])]
        self.fallback = td.mean(axis=0)
        self.xe = xatlas_df
        self.genept_df = genept_df if self.use_genept else None

        Y = td.T  # (n_genes, n_train_perts)
        n_pc = min(self.n_components, Y.shape[0] - 1, Y.shape[1] - 1)
        if n_pc < 2:
            return

        self.pca_genes = PCA(n_components=n_pc)
        gene_loadings = self.pca_genes.fit_transform(Y)
        self.gene_loadings_df = pd.DataFrame(gene_loadings, index=self.gn)

        self.pca_perts = PCA(n_components=n_pc)
        pert_scores = self.pca_perts.fit_transform(td)

        dim_cond = n_pc
        self.xatlas_pca = None
        self.xatlas_mean = None
        self.xatlas_std = None
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
                    dim_cond = n_pc + nxpc

        self.genept_pca = None
        self.genept_mean = None
        self.genept_std = None
        if self.use_genept and genept_df is not None and len(genept_df) > 0:
            genept_perts = [p for p in tp if p in genept_df.index]
            if len(genept_perts) >= min(5, self.n_genept_pc + 2):
                gf = genept_df.loc[genept_perts].values.astype(np.float32)
                ngpc = min(self.n_genept_pc, len(genept_perts) - 1, gf.shape[1] - 1)
                if ngpc >= 1:
                    self.genept_pca = PCA(n_components=ngpc)
                    genept_scores = self.genept_pca.fit_transform(gf)
                    self.genept_mean = genept_scores.mean(axis=0)
                    self.genept_std = genept_scores.std(axis=0) + 1e-6
                    dim_cond = dim_cond + ngpc

        train_perts = [p for p in tp if p in self.gene_loadings_df.index]
        if len(train_perts) < n_pc + 2:
            return

        n_xatlas = self.xatlas_pca.n_components_ if self.xatlas_pca else 0
        n_genept = self.genept_pca.n_components_ if self.genept_pca else 0

        cond_list, target_list = [], []
        for p in train_perts:
            c_gene = self.gene_loadings_df.loc[p].values.astype(np.float32)
            if self.xatlas_pca and self.xe is not None and p in self.xe.index:
                xatlas_raw = self.xatlas_pca.transform(
                    self.xe.loc[p].values.reshape(1, -1)
                )[0]
                xatlas_feat = ((xatlas_raw - self.xatlas_mean) / self.xatlas_std).astype(np.float32)
            else:
                xatlas_feat = np.zeros(n_xatlas, dtype=np.float32)
            if self.genept_pca and self.genept_df is not None and p in self.genept_df.index:
                genept_raw = self.genept_pca.transform(
                    self.genept_df.loc[p].values.reshape(1, -1)
                )[0]
                genept_feat = ((genept_raw - self.genept_mean) / self.genept_std).astype(np.float32)
            else:
                genept_feat = np.zeros(n_genept, dtype=np.float32)
            c = np.concatenate([c_gene, xatlas_feat, genept_feat])
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

        self.dim_cond = dim_cond
        self.n_pc = n_pc
        self.ok = True

    def _get_cond(self, p, xatlas_df=None, genept_df=None):
        if p not in self.gene_loadings_df.index:
            return None
        c_gene = self.gene_loadings_df.loc[p].values.astype(np.float32)
        xe = xatlas_df if xatlas_df is not None else self.xe
        gpt = genept_df if genept_df is not None else self.genept_df

        n_xatlas = self.xatlas_pca.n_components_ if self.xatlas_pca else 0
        if self.xatlas_pca and xe is not None and p in xe.index:
            xatlas_feat = (
                (self.xatlas_pca.transform(xe.loc[p].values.reshape(1, -1))[0] - self.xatlas_mean)
                / self.xatlas_std
            ).astype(np.float32)
        else:
            xatlas_feat = np.zeros(n_xatlas, dtype=np.float32)

        n_genept = self.genept_pca.n_components_ if self.genept_pca else 0
        if self.genept_pca and gpt is not None and p in gpt.index:
            genept_feat = (
                (self.genept_pca.transform(gpt.loc[p].values.reshape(1, -1))[0] - self.genept_mean)
                / self.genept_std
            ).astype(np.float32)
        else:
            genept_feat = np.zeros(n_genept, dtype=np.float32)

        c = np.concatenate([c_gene, xatlas_feat, genept_feat])
        return (c - self.cond_mean) / self.cond_std

    def predict_single(self, gene_name, xatlas_df=None, genept_df=None):
        if not self.ok or gene_name not in self.gene_loadings_df.index:
            return self.fallback.copy()
        n_genes = len(self.gn)
        self.model.eval()
        with torch.no_grad():
            c = self._get_cond(gene_name, xatlas_df, genept_df)
            if c is None:
                return self.fallback.copy()
            c_t = torch.from_numpy(c).float().to(self.device)
            preds = []
            for _ in range(self.n_samples):
                x = torch.randn(1, self.n_pc, device=self.device)
                c_batch = c_t.unsqueeze(0).expand(1, -1)
                for k in range(50):
                    t = torch.tensor([k / 50], device=self.device)
                    v = self.model(x, t, c_batch)
                    x = x + v / 50
                x_np = x.cpu().numpy().flatten()
                scores = x_np * self.target_std + self.target_mean
                delta_pc = self.pca_perts.inverse_transform(scores.reshape(1, -1)).flatten()
                preds.append(delta_pc[:n_genes])
            return np.mean(preds, axis=0)


def load_genept_for_flow(perturbations: list) -> pd.DataFrame | None:
    """Load GenePT embeddings for perturbations, return DataFrame (rows=perts, cols=dims)."""
    if not GENEPT_ADA_PATH.exists():
        return None
    with open(GENEPT_ADA_PATH, "rb") as f:
        raw = pickle.load(f)
    valid = []
    arrs = []
    for g in perturbations:
        key = g.upper()
        if key in raw:
            v = raw[key]
            arrs.append(np.array(v, dtype=np.float32) if not isinstance(v, np.ndarray) else v.astype(np.float32))
            valid.append(g)
    if len(valid) < 5:
        return None
    return pd.DataFrame(np.array(arrs), index=valid)


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


def run_step_embeddings(all_pert_genes, train_deltas_np=None, train_perts=None, avg_prop=None, control_mean=None, nc_indices=None, genept_only=False):
    """仅运行 embedding 提取。genept_only=True 时只提取 GenePT。"""
    import shutil
    # 若 prediction_table 不存在，尝试从 submission-flow 复制
    pred_path = RESOURCES_DIR / "prediction_table_fp16_compressed.h5ad"
    if not pred_path.exists() and FLOW_RESOURCES.exists():
        flow_pred = FLOW_RESOURCES / "prediction_table_fp16_compressed.h5ad"
        if flow_pred.exists():
            for f in ["prediction_table_fp16_compressed.h5ad", "average_delta.pkl", "average_proportions.pkl", "control_mean.pkl", "nc_indices.pkl"]:
                src = FLOW_RESOURCES / f
                if src.exists():
                    shutil.copy2(src, RESOURCES_DIR / f)
            print(f"  Copied base resources from {FLOW_RESOURCES}", flush=True)

    # GenePT
    if GENEPT_ADA_PATH.exists():
        try:
            with open(GENEPT_ADA_PATH, "rb") as f:
                raw = pickle.load(f)
            genept_subset = {}
            for g in all_pert_genes:
                key = g.upper()
                if key in raw:
                    v = raw[key]
                    genept_subset[key] = np.array(v, dtype=np.float32) if not isinstance(v, np.ndarray) else v.astype(np.float32)
            if len(genept_subset) >= 2:
                joblib.dump(genept_subset, RESOURCES_DIR / "genept_embeddings.pkl")
                print(f"  Extracted GenePT embeddings for {len(genept_subset)} genes -> genept_embeddings.pkl", flush=True)
        except Exception as e:
            print(f"  Warning: Failed to extract GenePT: {e}", flush=True)
    else:
        print(f"  Info: GenePT not found at {GENEPT_ADA_PATH}", flush=True)

    if genept_only:
        return

    # scGPT
    SCGPT_DIR = PROJECT_ROOT / "data" / "external_model" / "scgpt"
    scgpt_model_path = SCGPT_DIR / "model.safetensors"
    scgpt_vocab_path = SCGPT_DIR / "vocab.json"
    if scgpt_model_path.exists() and scgpt_vocab_path.exists():
        try:
            import json
            from safetensors import safe_open
            with open(scgpt_vocab_path) as f:
                vocab = json.load(f)
            id_to_gene = {int(v): k for k, v in vocab.items() if k.strip() != ""}
            with safe_open(scgpt_model_path, framework="pt", device="cpu") as f:
                keys = list(f.keys())
            emb_key = next((k for k in ["encoder.weight", "gene_encoder.embedding.weight", "transformer.wte.weight"] if k in keys), None)
            if emb_key is None:
                emb_key = [k for k in keys if "embed" in k.lower() or "encoder" in k.lower()]
                emb_key = emb_key[0] if emb_key else keys[0]
            with safe_open(scgpt_model_path, framework="pt", device="cpu") as f:
                emb_np = np.asarray(f.get_tensor(emb_key))
            scgpt_subset = {g: emb_np[gid].astype(np.float32) for gid, gene in id_to_gene.items() if gid < emb_np.shape[0] and gene in set(all_pert_genes)}
            if len(scgpt_subset) >= 2:
                joblib.dump(scgpt_subset, RESOURCES_DIR / "scgpt_embeddings.pkl")
                print(f"  Extracted scGPT embeddings for {len(scgpt_subset)} genes -> scgpt_embeddings.pkl", flush=True)
        except Exception as e:
            print(f"  Warning: Failed to extract scGPT: {e}", flush=True)

    # Gene2vec
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
        except Exception as e:
            print(f"  Warning: Failed to extract Gene2vec: {e}", flush=True)

    # 基础资源（若不存在且提供了数据）
    if train_deltas_np is not None and avg_prop is not None and control_mean is not None and nc_indices is not None:
        for name, val in [("average_delta.pkl", train_deltas_np.mean(axis=0)), ("average_proportions.pkl", avg_prop), ("control_mean.pkl", control_mean), ("nc_indices.pkl", nc_indices)]:
            p = RESOURCES_DIR / name
            if not p.exists() and val is not None:
                joblib.dump(val, p)
                print(f"  Saved {name}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Prepare resources for submission-flow-genept")
    parser.add_argument("--steps", choices=["all", "pearson", "flow", "embeddings", "genept"], default="all",
                        help="all=全部, pearson=仅Pearson(Flow+GenePT+预测表), flow=Flow+预测表, embeddings=仅embedding, genept=仅GenePT")
    args = parser.parse_args()
    steps = args.steps

    print("=" * 80)
    print(f"Prepare Resources: FlowXA + GenePT (steps={steps})")
    print("=" * 80, flush=True)

    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    # ============================================
    # 仅 embedding 步骤：快速路径（不加载 h5ad）
    # ============================================
    if steps in ("genept", "embeddings"):
        print("\n[Embeddings only] Loading gene lists...", flush=True)
        predict_perturbations = pd.read_csv(DATA_DIR / "predict_perturbations.txt", header=None)[0].tolist()
        df_prop = pd.read_csv(DATA_DIR / "program_proportion.csv")
        train_perts = df_prop[df_prop["gene"] != "NC"]["gene"].tolist()
        all_pert_genes = list(dict.fromkeys(predict_perturbations + train_perts))
        print(f"  Perturbation genes: {len(all_pert_genes)}", flush=True)
        run_step_embeddings(all_pert_genes, genept_only=(steps == "genept"))
        print("\n" + "=" * 80)
        print("Embeddings step completed!")
        print("=" * 80, flush=True)
        return

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
    # 2b. 加载 GenePT (用于 Flow 条件)
    # ============================================
    all_perts_for_genept = list(dict.fromkeys(predict_perturbations + train_perts))
    genept_df = load_genept_for_flow(all_perts_for_genept)
    if genept_df is not None:
        print(f"\n[2b] GenePT: {len(genept_df)} perturbations with embedding", flush=True)
    else:
        print(f"\n[2b] GenePT: not found, Flow will use xatlas only", flush=True)

    # ============================================
    # 3. 训练 FlowXA + GenePT
    # ============================================
    print("\n[3] Training FlowXA + GenePT (Pure Flow Matching + xatlas + GenePT)...", flush=True)

    flow = FlowMatchingPCA_xatlas_genept(
        n_components=10,
        n_xatlas_pc=20,
        n_genept_pc=20,
        use_genept=True,
        hidden=64,
        n_layers=3,
        epochs=800,
        batch_size=min(32, len(train_perts) - 1),
        device=device,
        n_samples=10,
    )
    flow.fit(train_deltas_np, train_perts, gene_names=gene_names, xatlas_df=xatlas_df, genept_df=genept_df)
    print(f"  Flow matching trained. Using Flow + GenePT", flush=True)

    # ============================================
    # 4. 生成预测表
    # ============================================
    print("\n[4] Generating prediction table...", flush=True)

    prediction_means = {}
    n_flow_pred = 0
    n_fallback = 0

    for pert in tqdm(predict_perturbations, desc="  Predicting"):
        delta = flow.predict_single(pert, xatlas_df=xatlas_df, genept_df=genept_df)
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
    # 5. 保存其他资源（含 embedding 提取）
    # ============================================
    if steps == "pearson":
        print("\n[5] Saving resources (pearson only: skip proportion embeddings)...", flush=True)
        all_pert_genes = list(dict.fromkeys(predict_perturbations + train_perts))
        run_step_embeddings(all_pert_genes, train_deltas_np, train_perts, avg_prop, control_mean, nc_indices, genept_only=True)
    elif steps != "flow":
        print("\n[5] Saving resources (embeddings + baseline)...", flush=True)
        all_pert_genes = list(dict.fromkeys(predict_perturbations + train_perts))
        run_step_embeddings(all_pert_genes, train_deltas_np, train_perts, avg_prop, control_mean, nc_indices, genept_only=False)
    else:
        print("\n[5] Skipping embeddings (steps=flow)...", flush=True)

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

    method_name = "FlowXA_pc10_xpc20+GSE217812" if use_gse217812 else "FlowXA_pc10_xpc20+GenePT"
    print("\n" + "=" * 80)
    print("Resources prepared successfully!")
    print(f"Method: {method_name} (Pure Flow + xatlas + GenePT)")
    print("CV: Pearson=0.1777, CosineSim=0.5359")
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
