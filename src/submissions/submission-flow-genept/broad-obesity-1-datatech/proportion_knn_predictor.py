"""
细胞比例 KNN 预测器（支持多源 Embedding + 可选 PCA）

使用 resources 中预提取的紧凑 pkl（scgpt_embeddings.pkl / gene2vec_embeddings.pkl），
不加载 scGPT 模型或完整 Gene2vec 大文件。

实验结论：scGPT+PCA32+K15 最优，Gene2vec+PCA32+K20 回退。缺失基因用全 0 向量。
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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


def _get_embed_dim(emb_dict):
    """从字典中获取 embedding 维度，用于缺失基因的零向量 fallback。"""
    if not emb_dict:
        return 200
    return len(next(iter(emb_dict.values())))


class ProportionKNNPredictor:
    """
    细胞比例 KNN 预测器。支持 Gene2vec / scGPT 等 embedding，可选 PCA 降维。
    """

    def __init__(
        self,
        k: int = 15,
        weights: str = "uniform",
        n_pca: int | None = None,
        embedding_source: str = "gene2vec",
    ):
        """
        k: KNN 邻居数
        weights: KNN 权重 ("uniform" / "distance")
        n_pca: PCA 组件数，None 表示不做 PCA
        embedding_source: "scgpt" | "gene2vec"，供 infer 加载对应嵌入
        """
        self.k = k
        self.weights = weights
        self.n_pca = n_pca
        self.embedding_source = embedding_source
        self.knn = None
        self.scaler = StandardScaler()
        self.pca = None
        self.train_genes = None
        self.cols = ["pre_adipo", "adipo", "lipo", "other"]

    def fit(
        self,
        train_genes: list,
        train_proportions: pd.DataFrame,
        emb_path_or_dict,
    ):
        """
        train_genes: 训练扰动基因名列表
        train_proportions: 含 gene, pre_adipo, adipo, lipo, other 的 DataFrame
        emb_path_or_dict: Gene2vec 文件路径或 {gene: vec} 字典（如 scGPT 预提取嵌入）
        """
        if isinstance(emb_path_or_dict, (str, Path)):
            emb = load_gene2vec_dict(emb_path_or_dict, train_genes)
        else:
            emb = emb_path_or_dict
            dim = _get_embed_dim(emb)

        dim = _get_embed_dim(emb)
        X = np.array([emb.get(g, np.zeros(dim, dtype=np.float32)) for g in train_genes])
        X = self.scaler.fit_transform(X)

        if self.n_pca is not None:
            n_comp = min(self.n_pca, X.shape[0] - 1, X.shape[1] - 1)
            n_comp = max(1, n_comp)
            self.pca = PCA(n_components=n_comp)
            X = self.pca.fit_transform(X)

        df = train_proportions[train_proportions["gene"].isin(train_genes)]
        df = df.set_index("gene").loc[train_genes].reset_index()
        y = df[self.cols].values

        self.knn = KNeighborsRegressor(
            n_neighbors=min(self.k, len(train_genes)), weights=self.weights
        )
        self.knn.fit(X, y)
        self.train_genes = train_genes
        return self

    def predict(self, test_genes: list, emb_path_or_dict) -> pd.DataFrame:
        """预测 test_genes 的比例。找不到 Embedding 的基因用全 0 向量。"""
        if isinstance(emb_path_or_dict, (str, Path)):
            emb = load_gene2vec_dict(emb_path_or_dict, test_genes)
        else:
            emb = emb_path_or_dict
            dim = _get_embed_dim(emb)

        dim = _get_embed_dim(emb)
        X = np.array([emb.get(g, np.zeros(dim, dtype=np.float32)) for g in test_genes])
        X = self.scaler.transform(X)

        if self.pca is not None:
            X = self.pca.transform(X)

        pred = self.knn.predict(X)
        pred_df = pd.DataFrame(
            {"gene": test_genes, **{c: pred[:, j] for j, c in enumerate(self.cols)}}
        )

        # 约束：pre_adipo + adipo + other = 1, lipo <= adipo
        s = pred_df[["pre_adipo", "adipo", "other"]].sum(axis=1).values
        s = np.maximum(s, 1e-10)
        for c in ["pre_adipo", "adipo", "other"]:
            pred_df[c] = pred_df[c].values / s
        pred_df["lipo"] = np.minimum(pred_df["lipo"].values, pred_df["adipo"].values)
        pred_df["lipo"] = np.maximum(pred_df["lipo"].values, 0)

        return pred_df
