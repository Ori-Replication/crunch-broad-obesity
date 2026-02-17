"""
Gene2vec KNN 细胞比例预测器

用于推理阶段：对未见基因，用训练集中 Gene2vec 最近邻的比例均值预测。
找不到 Embedding 的基因用全 0 向量表示。
5-fold CV 显示 K=10~20 时 L1≈0.092，优于 Average 0.098。
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor


def load_gene2vec_dict(path, gene_list=None):
    """
    加载 Gene2vec，返回 {gene: vec}。
    若 gene_list 给定，缺失的基因用零向量表示。
    """
    g2v = {}
    with open(path) as f:
        first = f.readline().strip().split()
        dim = int(first[1])
        for line in f:
            parts = line.strip().split()
            if len(parts) < dim + 1:
                continue
            gene = parts[0]
            vec = np.array([float(x) for x in parts[1 : dim + 1]], dtype=np.float32)
            g2v[gene] = vec
    if gene_list:
        zeros = np.zeros(dim, dtype=np.float32)
        for g in gene_list:
            if g not in g2v:
                g2v[g] = zeros.copy()
    return g2v


def _get_embed_dim(g2v):
    """从字典中获取 embedding 维度，用于缺失基因的零向量 fallback。"""
    if not g2v:
        return 200
    return len(next(iter(g2v.values())))


class ProportionKNNPredictor:
    """Gene2vec KNN 比例预测器。找不到 Embedding 的基因用全 0 表示。"""

    def __init__(self, k: int = 15, weights: str = "uniform"):
        self.k = k
        self.weights = weights
        self.knn = None
        self.scaler = StandardScaler()
        self.train_genes = None
        self.cols = ["pre_adipo", "adipo", "lipo", "other"]

    def fit(self, train_genes: list, train_proportions: pd.DataFrame, gene2vec_path_or_dict):
        """
        train_genes: 训练扰动基因名列表
        train_proportions: 含 gene, pre_adipo, adipo, lipo, other 的 DataFrame
        gene2vec_path_or_dict: Gene2vec 文件路径或 {gene: vec} 字典
        """
        if isinstance(gene2vec_path_or_dict, (str, Path)):
            g2v = load_gene2vec_dict(gene2vec_path_or_dict, train_genes)
        else:
            g2v = gene2vec_path_or_dict

        dim = _get_embed_dim(g2v)
        X = np.array([g2v.get(g, np.zeros(dim, dtype=np.float32)) for g in train_genes])
        X = self.scaler.fit_transform(X)

        df = train_proportions[train_proportions["gene"].isin(train_genes)]
        df = df.set_index("gene").loc[train_genes].reset_index()
        y = df[self.cols].values

        self.knn = KNeighborsRegressor(n_neighbors=min(self.k, len(train_genes)), weights=self.weights)
        self.knn.fit(X, y)
        self.train_genes = train_genes
        return self

    def predict(self, test_genes: list, gene2vec_path_or_dict) -> pd.DataFrame:
        """预测 test_genes 的比例。找不到 Embedding 的基因用全 0 向量。"""
        if isinstance(gene2vec_path_or_dict, (str, Path)):
            g2v = load_gene2vec_dict(gene2vec_path_or_dict, test_genes)
        else:
            g2v = gene2vec_path_or_dict

        dim = _get_embed_dim(g2v)
        X = np.array([g2v.get(g, np.zeros(dim, dtype=np.float32)) for g in test_genes])
        X = self.scaler.transform(X)

        pred = self.knn.predict(X)
        pred_df = pd.DataFrame({"gene": test_genes, **{c: pred[:, j] for j, c in enumerate(self.cols)}})

        # 约束：pre_adipo + adipo + other = 1, lipo <= adipo
        s = pred_df[["pre_adipo", "adipo", "other"]].sum(axis=1).values
        s = np.maximum(s, 1e-10)
        for c in ["pre_adipo", "adipo", "other"]:
            pred_df[c] = pred_df[c].values / s
        pred_df["lipo"] = np.minimum(pred_df["lipo"].values, pred_df["adipo"].values)
        pred_df["lipo"] = np.maximum(pred_df["lipo"].values, 0)

        return pred_df
