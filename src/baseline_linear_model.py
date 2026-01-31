"""
Baseline Linear Model Implementation
基于论文中的线性模型：Y ≈ GWP^T + b
实现五折交叉验证，避免测试集泄露
"""

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
from typing import Dict, Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

# PyTorch imports for CosineRidge
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. MainPyLinearModel will use sklearn Ridge instead.")


class BaselineLinearModel:
    """
    基线线性模型：Y ≈ GWP^T + b (双线性模型)
    
    - G: Gene Embeddings (通过训练集PCA得到)
    - P: Perturbation Embeddings (对于单基因扰动，取对应基因的G行)
    - W: 权重矩阵 (通过岭回归求解)
    - b: 截距向量 (训练数据的行均值)
    """
    
    def __init__(self, K: int = 10, lambda_reg: float = 0.1):
        """
        初始化模型
        
        Parameters:
        -----------
        K : int
            PCA主成分数量，默认10
        lambda_reg : float
            岭回归正则化系数，默认0.1
        """
        self.K = K
        self.lambda_reg = lambda_reg
        self.pca = None
        self.G = None  # Gene embeddings (n_genes × K)
        self.W = None  # Weight matrix (K × K)
        self.b = None  # Intercept vector (n_genes,)
        self.gene_to_idx = None  # 基因名到索引的映射


class SimpleLinearModel:
    """
    简单线性模型：delta = f(gene_embedding)
    
    - Gene Embeddings: 通过控制组（NC）细胞的SVD得到
    - Linear Mapping: gene_embedding → delta (通过岭回归)
    """
    
    def __init__(self, n_components: int = 50, lambda_reg: float = 0.1):
        """
        初始化模型
        
        Parameters:
        -----------
        n_components : int
            SVD主成分数量，默认50
        lambda_reg : float
            岭回归正则化系数，默认0.1
        """
        self.n_components = n_components
        self.lambda_reg = lambda_reg
        self.svd = None
        self.gene_embeddings = None  # (n_genes × n_components)
        self.gene_map = None  # 基因名到嵌入向量的映射
        self.control_mean = None  # 控制组均值 (n_genes,)
        self.delta_model = None  # 岭回归模型：gene_embedding → delta
        
    def fit(self, adata: sc.AnnData, Y_train: np.ndarray, 
            perturbation_genes: List[str], gene_names: List[str]) -> None:
        """
        训练模型
        
        Parameters:
        -----------
        adata : sc.AnnData
            单细胞数据（用于提取NC细胞）
        Y_train : np.ndarray
            训练集表达矩阵 (n_perturbations × n_genes)
        perturbation_genes : List[str]
            每个扰动对应的目标基因列表
        gene_names : List[str]
            基因名称列表
        """
        # 1. 提取控制组（NC）细胞
        nc_mask = adata.obs['gene'] == 'NC'
        nc_indices = np.where(nc_mask)[0]
        
        # 采样一部分NC细胞（避免内存问题）
        sample_size = min(len(nc_indices), 20000)
        sampled_nc_indices = np.random.choice(nc_indices, sample_size, replace=False)
        
        # 读取NC细胞表达数据
        X_nc_data = adata[sampled_nc_indices].X
        if hasattr(X_nc_data, 'toarray'):
            X_nc = X_nc_data.toarray()
        else:
            X_nc = np.asarray(X_nc_data)
        
        # 计算控制组均值
        self.control_mean = np.ravel(np.asarray(np.mean(X_nc, axis=0)))
        
        # 2. 使用NC细胞构建基因嵌入（SVD）
        self.svd = TruncatedSVD(n_components=self.n_components, random_state=42)
        self.svd.fit(X_nc)
        self.gene_embeddings = self.svd.components_.T  # (n_genes × n_components)
        
        # 构建基因名到嵌入的映射
        self.gene_map = {gene: self.gene_embeddings[i] for i, gene in enumerate(gene_names)}
        
        # 3. 准备训练数据：gene_embedding → delta
        X_train_reg = []
        y_train_reg = []
        
        for i, pert_gene in enumerate(perturbation_genes):
            if pert_gene in self.gene_map:
                # 获取该扰动的平均表达
                pert_mean = Y_train[i]
                # 计算delta
                delta = pert_mean - self.control_mean
                
                # 使用该基因的嵌入作为特征
                X_train_reg.append(self.gene_map[pert_gene])
                y_train_reg.append(delta)
        
        X_train_reg = np.array(X_train_reg)
        y_train_reg = np.array(y_train_reg)
        
        # 4. 训练岭回归模型：gene_embedding → delta
        self.delta_model = Ridge(alpha=self.lambda_reg)
        self.delta_model.fit(X_train_reg, y_train_reg)
        
    def predict_delta(self, perturbation_genes: List[str]) -> np.ndarray:
        """
        预测扰动效应（Delta）
        
        Parameters:
        -----------
        perturbation_genes : List[str]
            要预测的扰动基因列表
            
        Returns:
        --------
        delta_pred : np.ndarray
            预测的Delta矩阵 (n_perturbations × n_genes)
        """
        # 默认嵌入（如果基因不在训练集中）
        default_emb = np.mean(list(self.gene_map.values()), axis=0)
        
        X_pred = []
        for pert_gene in perturbation_genes:
            emb = self.gene_map.get(pert_gene, default_emb).reshape(1, -1)
            delta = self.delta_model.predict(emb).flatten()
            X_pred.append(delta)
        
        return np.array(X_pred)
    
    def predict(self, perturbation_genes: List[str]) -> np.ndarray:
        """
        预测完整表达矩阵
        
        Parameters:
        -----------
        perturbation_genes : List[str]
            要预测的扰动基因列表
            
        Returns:
        --------
        Y_pred : np.ndarray
            预测的表达矩阵 (n_perturbations × n_genes)
        """
        delta_pred = self.predict_delta(perturbation_genes)
        # Y_pred = control_mean + delta
        Y_pred = self.control_mean + delta_pred
        return Y_pred


class CosineRidge(nn.Module):
    """
    CosineRidge模型：使用余弦相似度损失 + L2正则化
    完全复现 main.py 中的实现
    """
    def __init__(self, in_dim, out_dim, alpha=10.0):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.alpha = alpha  # L2 正则系数

    def forward(self, x):
        return self.W(x)

    def loss(self, y_pred, y_true):
        # 中心化（这是关键！对齐 Pearson）
        y_pred_c = y_pred - y_pred.mean(dim=1, keepdim=True)
        y_true_c = y_true - y_true.mean(dim=1, keepdim=True)

        cos_sim = nn.functional.cosine_similarity(y_pred_c, y_true_c, dim=1)
        cosine_loss = 1.0 - cos_sim.mean()

        # L2 正则（Ridge）
        l2 = torch.sum(self.W.weight ** 2)
        return cosine_loss + self.alpha * l2


class PearsonSOTAModel:
    """
    完全复现 main_pearson_sota.py 中的线性模型实现
    
    关键特点（与main_pearson_sota.py完全一致）：
    1. 使用 NC 细胞的 SVD 得到基因嵌入（n_components=50）
    2. 使用 Ridge 训练 delta 预测（alpha=10.0，不是CosineRidge）
    3. 使用 Ridge 训练程序比例预测（alpha=1.0，不是0.001）
    4. 训练时每个扰动采样最多500个细胞
    5. 保存control_cells_sample（前1000个NC细胞）用于预测时采样
    6. 预测时从control_cells_sample中随机采样并应用delta
    """
    
    def __init__(self, n_components: int = 50, 
                 delta_alpha: float = 10.0,
                 prop_alpha: float = 1.0):
        """
        初始化模型
        
        Parameters:
        -----------
        n_components : int
            SVD主成分数量，默认50
        delta_alpha : float
            Delta模型的L2正则化系数，默认10.0
        prop_alpha : float
            程序比例模型的L2正则化系数，默认1.0（与main_pearson_sota.py一致）
        """
        self.n_components = n_components
        self.delta_alpha = delta_alpha
        self.prop_alpha = prop_alpha
        
        self.svd = None
        self.gene_embeddings = None
        self.gene_map = None
        self.control_mean = None
        self.control_cells_sample = None  # 保存前1000个NC细胞用于预测
        self.delta_model = None  # Ridge模型
        self.prop_model = None  # Ridge模型用于预测程序比例
        
    def fit(self, adata: sc.AnnData, 
            perturbation_genes: List[str],
            gene_names: List[str],
            proportion_df: Optional[pd.DataFrame] = None) -> None:
        """
        训练模型（完全复现 main_pearson_sota.py 的 train 函数）
        
        Parameters:
        -----------
        adata : sc.AnnData
            单细胞数据（用于提取NC细胞和扰动数据）
        perturbation_genes : List[str]
            训练集中的扰动基因列表
        gene_names : List[str]
            基因名称列表
        proportion_df : pd.DataFrame, optional
            程序比例数据框，包含 'gene', 'pre_adipo', 'adipo', 'lipo', 'other' 列
        """
        # 1. 提取控制组（NC）数据用于构建基线和基因嵌入
        print("Extracting Control (NC) data...")
        nc_indices = np.where(adata.obs["gene"] == "NC")[0]
        
        # 随机采样一部分 NC 细胞（最多20000个，与main_pearson_sota.py一致）
        sample_size = min(len(nc_indices), 20000)
        sampled_nc_indices = np.random.choice(nc_indices, sample_size, replace=False)
        
        # 安全读取数据
        X_nc_data = adata[sampled_nc_indices].X
        if hasattr(X_nc_data, "toarray"):
            X_nc = X_nc_data.toarray()
        else:
            X_nc = np.asarray(X_nc_data)
        
        # 计算控制组均值
        self.control_mean = np.ravel(np.asarray(np.mean(X_nc, axis=0)))
        
        # 保存前1000个NC细胞用于预测（与main_pearson_sota.py一致）
        self.control_cells_sample = X_nc[:1000]
        
        # 2. 构建基因嵌入（Gene Embeddings）
        print("Computing Gene Embeddings via SVD...")
        self.svd = TruncatedSVD(n_components=self.n_components, random_state=42)
        self.svd.fit(X_nc)
        self.gene_embeddings = self.svd.components_.T  # Shape: (n_genes, n_components)
        
        # 构建基因名到嵌入的映射
        self.gene_map = {name: self.gene_embeddings[i] for i, name in enumerate(gene_names)}
        
        # 3. 准备线性回归训练数据
        print("Preparing regression training data...")
        X_train_reg = []
        y_train_reg = []
        X_train_prop = []
        y_train_prop = []
        
        unique_perts = list(set(perturbation_genes))
        
        for pert in unique_perts:
            if pert not in self.gene_map:
                continue
            
            pert_indices = np.where(adata.obs["gene"] == pert)[0]
            if len(pert_indices) < 5:
                continue
            
            # 每个扰动采样最多500个细胞（与main_pearson_sota.py一致）
            idx_to_load = pert_indices[:500]
            
            # 安全读取数据
            X_pert_data = adata[idx_to_load].X
            if hasattr(X_pert_data, "toarray"):
                X_pert = X_pert_data.toarray()
            else:
                X_pert = np.asarray(X_pert_data)
            
            # 计算扰动的平均表达和delta
            pert_mean = np.ravel(np.asarray(np.mean(X_pert, axis=0)))
            delta = pert_mean - self.control_mean
            
            X_train_reg.append(self.gene_map[pert])
            y_train_reg.append(delta)
            
            # 如果有程序比例数据，也准备训练数据
            if proportion_df is not None:
                row = proportion_df[proportion_df['gene'] == pert]
                if len(row) > 0:
                    targets = row[['pre_adipo', 'adipo', 'lipo', 'other']].values[0].astype(float)
                    X_train_prop.append(self.gene_map[pert])
                    y_train_prop.append(targets)
        
        X_train_reg = np.array(X_train_reg)
        y_train_reg = np.array(y_train_reg)
        
        print(f"X_train_reg shape: {X_train_reg.shape}")
        print(f"y_train_reg shape: {y_train_reg.shape}")
        
        # 4. 训练模型（使用Ridge，不是CosineRidge）
        print("Training Linear Models (Ridge)...")
        
        # Delta模型：使用Ridge(alpha=10.0)
        print(f"Training Delta Model (Ridge, alpha={self.delta_alpha})...")
        self.delta_model = Ridge(alpha=self.delta_alpha)
        self.delta_model.fit(X_train_reg, y_train_reg)
        
        # 训练程序比例模型：使用Ridge(alpha=1.0)
        if len(X_train_prop) > 0:
            X_train_prop = np.array(X_train_prop)
            y_train_prop = np.array(y_train_prop)
            print(f"Training Proportion Model (Ridge, alpha={self.prop_alpha})...")
            self.prop_model = Ridge(alpha=self.prop_alpha)
            self.prop_model.fit(X_train_prop, y_train_prop)
        else:
            self.prop_model = None
            print("Warning: No proportion data available, skipping proportion model training.")
        
        print("Linear models trained.")
    
    def predict_delta(self, perturbation_genes: List[str]) -> np.ndarray:
        """
        预测扰动效应（Delta）
        
        Parameters:
        -----------
        perturbation_genes : List[str]
            要预测的扰动基因列表
            
        Returns:
        --------
        delta_pred : np.ndarray
            预测的Delta矩阵 (n_perturbations × n_genes)
        """
        # 默认嵌入（如果基因不在训练集中）
        default_emb = np.mean(list(self.gene_map.values()), axis=0)
        
        X_pred = []
        for pert_gene in perturbation_genes:
            emb = self.gene_map.get(pert_gene, default_emb).reshape(1, -1)
            delta = self.delta_model.predict(emb).flatten()
            X_pred.append(delta)
        
        return np.array(X_pred)
    
    def predict_proportion(self, perturbation_genes: List[str]) -> Optional[np.ndarray]:
        """
        预测程序比例
        
        Parameters:
        -----------
        perturbation_genes : List[str]
            要预测的扰动基因列表
            
        Returns:
        --------
        prop_pred : np.ndarray or None
            预测的程序比例矩阵 (n_perturbations × 4)，列顺序为 [pre_adipo, adipo, lipo, other]
            如果模型未训练则返回None
        """
        if self.prop_model is None:
            return None
        
        default_emb = np.mean(list(self.gene_map.values()), axis=0)
        
        X_pred = []
        for pert_gene in perturbation_genes:
            emb = self.gene_map.get(pert_gene, default_emb).reshape(1, -1)
            prop = self.prop_model.predict(emb).flatten()
            # 修正比例（非负 + 归一化）
            prop = np.clip(prop, 0, None)
            if prop.sum() > 0:
                prop /= prop.sum()
            else:
                prop = np.array([0.25, 0.25, 0.25, 0.25])  # 默认均匀分布
            X_pred.append(prop)
        
        return np.array(X_pred)
    
    def predict(self, perturbation_genes: List[str], 
                cells_per_perturbation: int = 100) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        预测完整表达矩阵（生成单细胞数据）
        完全复现main_pearson_sota.py的infer函数逻辑
        
        Parameters:
        -----------
        perturbation_genes : List[str]
            要预测的扰动基因列表
        cells_per_perturbation : int
            每个扰动生成的细胞数量，默认100
            
        Returns:
        --------
        cells_pred : np.ndarray
            预测的细胞表达矩阵 (n_cells × n_genes)
        prop_pred : np.ndarray or None
            预测的程序比例矩阵 (n_perturbations × 4)
        """
        if self.control_cells_sample is None:
            raise ValueError("Model not trained properly. control_cells_sample is None.")
        
        # 预测delta和比例
        delta_pred = self.predict_delta(perturbation_genes)
        prop_pred = self.predict_proportion(perturbation_genes)
        
        # 生成单细胞数据（与main_pearson_sota.py一致）
        X_pred_all = []
        
        for i, pert in enumerate(perturbation_genes):
            # 从control_cells_sample中随机采样
            indices = np.random.choice(len(self.control_cells_sample), 
                                      cells_per_perturbation, 
                                      replace=True)
            base_cells = self.control_cells_sample[indices]
            
            # 应用delta：Cell_new = Cell_control + Delta
            generated_cells = base_cells + delta_pred[i]
            
            # 修正负值（表达量不能为负）
            generated_cells = np.clip(generated_cells, 0, None)
            
            X_pred_all.append(generated_cells)
        
        cells_pred = np.vstack(X_pred_all)
        
        return cells_pred, prop_pred
    
    def predict_mean(self, perturbation_genes: List[str]) -> np.ndarray:
        """
        预测平均表达矩阵（用于评估）
        
        Parameters:
        -----------
        perturbation_genes : List[str]
            要预测的扰动基因列表
            
        Returns:
        --------
        Y_pred : np.ndarray
            预测的平均表达矩阵 (n_perturbations × n_genes)
        """
        delta_pred = self.predict_delta(perturbation_genes)
        # Y_pred = control_mean + delta
        Y_pred = self.control_mean + delta_pred
        return Y_pred


class MainPyLinearModel:
    """
    完全复现 main.py 中的线性模型实现
    
    关键特点：
    1. 使用 NC 细胞的 SVD 得到基因嵌入（n_components=50）
    2. 使用 CosineRidge 训练 delta 预测（alpha=10.0, 500 epochs）
    3. 使用 Ridge 训练程序比例预测（alpha=0.001）
    4. 预测时从 NC 细胞中按比例采样并应用 delta
    """
    
    def __init__(self, n_components: int = 50, 
                 cosine_alpha: float = 10.0,
                 prop_alpha: float = 0.001,
                 n_epochs: int = 500,
                 learning_rate: float = 1e-2,
                 use_torch: bool = True):
        """
        初始化模型
        
        Parameters:
        -----------
        n_components : int
            SVD主成分数量，默认50
        cosine_alpha : float
            CosineRidge的L2正则化系数，默认10.0
        prop_alpha : float
            程序比例模型的L2正则化系数，默认0.001
        n_epochs : int
            CosineRidge训练的epoch数，默认500
        learning_rate : float
            优化器学习率，默认1e-2
        use_torch : bool
            是否使用PyTorch的CosineRidge，如果False则使用sklearn Ridge
        """
        self.n_components = n_components
        self.cosine_alpha = cosine_alpha
        self.prop_alpha = prop_alpha
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.use_torch = use_torch and TORCH_AVAILABLE
        
        self.svd = None
        self.gene_embeddings = None
        self.gene_map = None
        self.control_mean = None
        self.delta_model = None  # CosineRidge 或 Ridge
        self.delta_weight = None  # 如果使用PyTorch，保存权重矩阵
        self.prop_model = None  # Ridge模型用于预测程序比例
        
    def fit(self, adata: sc.AnnData, 
            perturbation_genes: List[str],
            gene_names: List[str],
            proportion_df: Optional[pd.DataFrame] = None) -> None:
        """
        训练模型（完全复现 main.py 的 train 函数）
        
        Parameters:
        -----------
        adata : sc.AnnData
            单细胞数据（用于提取NC细胞和扰动数据）
        perturbation_genes : List[str]
            训练集中的扰动基因列表
        gene_names : List[str]
            基因名称列表
        proportion_df : pd.DataFrame, optional
            程序比例数据框，包含 'gene', 'pre_adipo', 'adipo', 'lipo', 'other' 列
        """
        # 1. 提取控制组（NC）数据用于构建基线和基因嵌入
        print("Extracting Control (NC) data...")
        nc_indices = np.where(adata.obs["gene"] == "NC")[0]
        
        # 随机采样一部分 NC 细胞（最多20000个）
        sample_size = min(len(nc_indices), 20000)
        sampled_nc_indices = np.random.choice(nc_indices, sample_size, replace=False)
        
        # 安全读取数据
        X_nc_data = adata[sampled_nc_indices].X
        if hasattr(X_nc_data, "toarray"):
            X_nc = X_nc_data.toarray()
        else:
            X_nc = np.asarray(X_nc_data)
        
        # 计算控制组均值
        self.control_mean = np.ravel(np.asarray(np.mean(X_nc, axis=0)))
        
        # 2. 构建基因嵌入（Gene Embeddings）
        print("Computing Gene Embeddings via SVD...")
        self.svd = TruncatedSVD(n_components=self.n_components, random_state=42)
        self.svd.fit(X_nc)
        self.gene_embeddings = self.svd.components_.T  # Shape: (n_genes, n_components)
        
        # 构建基因名到嵌入的映射
        self.gene_map = {name: self.gene_embeddings[i] for i, name in enumerate(gene_names)}
        
        # 3. 准备线性回归训练数据
        print("Preparing regression training data...")
        X_train_reg = []
        y_train_reg = []
        X_train_prop = []
        y_train_prop = []
        
        unique_perts = list(set(perturbation_genes))
        
        for pert in unique_perts:
            if pert not in self.gene_map:
                continue
            
            pert_indices = np.where(adata.obs["gene"] == pert)[0]
            if len(pert_indices) < 5:
                continue
            
            # 安全读取数据
            X_pert_data = adata[pert_indices].X
            if hasattr(X_pert_data, "toarray"):
                X_pert = X_pert_data.toarray()
            else:
                X_pert = X_pert_data
            
            # 计算扰动的平均表达和delta
            pert_mean = np.ravel(np.asarray(np.mean(X_pert, axis=0)))
            delta = pert_mean - self.control_mean
            
            X_train_reg.append(self.gene_map[pert])
            y_train_reg.append(delta)
            
            # 如果有程序比例数据，也准备训练数据
            if proportion_df is not None:
                row = proportion_df[proportion_df['gene'] == pert]
                if len(row) > 0:
                    targets = row[['pre_adipo', 'adipo', 'lipo', 'other']].values[0].astype(float)
                    X_train_prop.append(self.gene_map[pert])
                    y_train_prop.append(targets)
        
        X_train_reg = np.array(X_train_reg)
        y_train_reg = np.array(y_train_reg)
        
        print(f"X_train_reg shape: {X_train_reg.shape}")
        print(f"y_train_reg shape: {y_train_reg.shape}")
        
        # 4. 训练模型
        print("Training Linear Models...")
        
        if self.use_torch:
            # 使用 CosineRidge（PyTorch）
            print(f"Training CosineRidge (alpha={self.cosine_alpha}, epochs={self.n_epochs})...")
            X_torch = torch.tensor(X_train_reg, dtype=torch.float32)
            Y_torch = torch.tensor(y_train_reg, dtype=torch.float32)
            
            model = CosineRidge(
                in_dim=X_torch.shape[1],
                out_dim=Y_torch.shape[1],
                alpha=self.cosine_alpha
            )
            
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
            
            for epoch in range(self.n_epochs):
                optimizer.zero_grad()
                pred = model(X_torch)
                loss = model.loss(pred, Y_torch)
                loss.backward()
                optimizer.step()
                
                if epoch % 50 == 0:
                    print(f"[CosineRidge] Epoch {epoch}, loss={loss.item():.4f}")
            
            # 保存权重矩阵
            self.delta_weight = model.W.weight.detach().cpu().numpy()
            self.delta_model = None  # 不使用sklearn模型
        else:
            # 使用 sklearn Ridge（近似）
            print(f"Training Ridge Regression (alpha={self.cosine_alpha})...")
            self.delta_model = Ridge(alpha=self.cosine_alpha)
            self.delta_model.fit(X_train_reg, y_train_reg)
            self.delta_weight = None
        
        # 训练程序比例模型
        if len(X_train_prop) > 0:
            X_train_prop = np.array(X_train_prop)
            y_train_prop = np.array(y_train_prop)
            print(f"Training Proportion Model (alpha={self.prop_alpha})...")
            self.prop_model = Ridge(alpha=self.prop_alpha)
            self.prop_model.fit(X_train_prop, y_train_prop)
        else:
            self.prop_model = None
            print("Warning: No proportion data available, skipping proportion model training.")
        
        print("Linear models trained.")
    
    def predict_delta(self, perturbation_genes: List[str]) -> np.ndarray:
        """
        预测扰动效应（Delta）
        
        Parameters:
        -----------
        perturbation_genes : List[str]
            要预测的扰动基因列表
            
        Returns:
        --------
        delta_pred : np.ndarray
            预测的Delta矩阵 (n_perturbations × n_genes)
        """
        # 默认嵌入（如果基因不在训练集中）
        default_emb = np.mean(list(self.gene_map.values()), axis=0)
        
        X_pred = []
        for pert_gene in perturbation_genes:
            emb = self.gene_map.get(pert_gene, default_emb).reshape(1, -1)
            
            if self.use_torch and self.delta_weight is not None:
                # 使用保存的权重矩阵
                delta = (emb @ self.delta_weight.T).flatten()
            elif self.delta_model is not None:
                # 使用sklearn模型
                delta = self.delta_model.predict(emb).flatten()
            else:
                raise ValueError("Model not trained properly.")
            
            X_pred.append(delta)
        
        return np.array(X_pred)
    
    def predict_proportion(self, perturbation_genes: List[str]) -> Optional[np.ndarray]:
        """
        预测程序比例
        
        Parameters:
        -----------
        perturbation_genes : List[str]
            要预测的扰动基因列表
            
        Returns:
        --------
        prop_pred : np.ndarray or None
            预测的程序比例矩阵 (n_perturbations × 4)，列顺序为 [pre_adipo, adipo, lipo, other]
            如果模型未训练则返回None
        """
        if self.prop_model is None:
            return None
        
        default_emb = np.mean(list(self.gene_map.values()), axis=0)
        
        X_pred = []
        for pert_gene in perturbation_genes:
            emb = self.gene_map.get(pert_gene, default_emb).reshape(1, -1)
            prop = self.prop_model.predict(emb).flatten()
            # 修正比例（非负 + 归一化）
            prop = np.clip(prop, 0, None)
            if prop.sum() > 0:
                prop /= prop.sum()
            X_pred.append(prop)
        
        return np.array(X_pred)
    
    def predict(self, perturbation_genes: List[str]) -> np.ndarray:
        """
        预测完整表达矩阵（平均表达）
        
        Parameters:
        -----------
        perturbation_genes : List[str]
            要预测的扰动基因列表
            
        Returns:
        --------
        Y_pred : np.ndarray
            预测的表达矩阵 (n_perturbations × n_genes)
        """
        delta_pred = self.predict_delta(perturbation_genes)
        # Y_pred = control_mean + delta
        Y_pred = self.control_mean + delta_pred
        return Y_pred


class BaselineLinearModel:
    """
    基线线性模型：Y ≈ GWP^T + b (双线性模型)
    
    - G: Gene Embeddings (通过训练集PCA得到)
    - P: Perturbation Embeddings (对于单基因扰动，取对应基因的G行)
    - W: 权重矩阵 (通过岭回归求解)
    - b: 截距向量 (训练数据的行均值)
    """
    
    def __init__(self, K: int = 10, lambda_reg: float = 0.1):
        """
        初始化模型
        
        Parameters:
        -----------
        K : int
            PCA主成分数量，默认10
        lambda_reg : float
            岭回归正则化系数，默认0.1
        """
        self.K = K
        self.lambda_reg = lambda_reg
        self.pca = None
        self.G = None  # Gene embeddings (n_genes × K)
        self.W = None  # Weight matrix (K × K)
        self.b = None  # Intercept vector (n_genes,)
        self.gene_to_idx = None  # 基因名到索引的映射
        
    def fit(self, Y_train: np.ndarray, gene_names: List[str], 
            perturbation_genes: List[str]) -> None:
        """
        训练模型
        
        Parameters:
        -----------
        Y_train : np.ndarray
            训练集表达矩阵 (n_perturbations × n_genes)
        gene_names : List[str]
            基因名称列表
        perturbation_genes : List[str]
            每个扰动对应的目标基因列表 (长度 = n_perturbations)
        """
        n_perturbations, n_genes = Y_train.shape
        
        # 1. 计算截距 b (每行的均值)
        self.b = Y_train.mean(axis=0)  # (n_genes,)
        
        # 2. 中心化数据
        Y_centered = Y_train - self.b  # (n_perturbations × n_genes)
        
        # 3. 对训练集进行PCA得到G矩阵
        self.pca = PCA(n_components=self.K)
        # PCA在基因维度上进行，转置后每行是一个基因
        self.G = self.pca.fit_transform(Y_centered.T)  # (n_genes × K)
        
        # 4. 构建P矩阵 (Perturbation Embeddings)
        # 对于单基因扰动，P矩阵中对应行取G矩阵中该基因的行
        self.gene_to_idx = {gene: idx for idx, gene in enumerate(gene_names)}
        P = np.zeros((n_perturbations, self.K))  # (n_perturbations × K)
        
        for i, pert_gene in enumerate(perturbation_genes):
            if pert_gene in self.gene_to_idx:
                gene_idx = self.gene_to_idx[pert_gene]
                P[i] = self.G[gene_idx]
            else:
                # 如果扰动基因不在训练集中，使用零向量或均值
                # 这里使用零向量
                P[i] = np.zeros(self.K)
        
        # 5. 使用岭回归求解W矩阵
        # 根据论文公式：Y ≈ GWP^T + b
        # 其中 Y: (n_perturbations × n_genes)
        #     G: (n_genes × K)
        #     W: (K × K)
        #     P: (n_perturbations × K)
        #     P^T: (K × n_perturbations)
        #
        # 注意：GWP^T = (n_genes × K) @ (K × K) @ (K × n_perturbations) = (n_genes × n_perturbations)
        # 所以需要转置：Y^T ≈ GWP^T，即 Y ≈ (GWP^T)^T = P W^T G^T
        #
        # 但根据论文中的求解公式（baseline.md第48行）：
        # W = (G^T G + λI)^(-1) G^T (Y_train - b) P (P^T P + λI)^(-1)
        # 这里 (Y_train - b) 需要转置为 (n_genes × n_perturbations)
        
        GtG = self.G.T @ self.G  # (K × K)
        PtP = P.T @ P  # (K × K)
        
        # 添加正则化项
        GtG_reg = GtG + self.lambda_reg * np.eye(self.K)
        PtP_reg = PtP + self.lambda_reg * np.eye(self.K)
        
        # 计算逆矩阵
        GtG_inv = np.linalg.inv(GtG_reg)
        PtP_inv = np.linalg.inv(PtP_reg)
        
        # 根据论文公式求解W
        # W = (G^T G + λI)^(-1) G^T (Y_train - b)^T P (P^T P + λI)^(-1)
        # 注意：Y_centered = Y_train - b，维度是 (n_perturbations × n_genes)
        # 需要转置为 (n_genes × n_perturbations)
        self.W = GtG_inv @ self.G.T @ Y_centered.T @ P @ PtP_inv  # (K × K)
        
    def predict(self, perturbation_genes: List[str]) -> np.ndarray:
        """
        预测扰动效果
        
        Parameters:
        -----------
        perturbation_genes : List[str]
            要预测的扰动基因列表
            
        Returns:
        --------
        Y_pred : np.ndarray
            预测的表达矩阵 (n_perturbations × n_genes)
        """
        n_perturbations = len(perturbation_genes)
        n_genes = self.G.shape[0]
        
        # 构建P矩阵
        P = np.zeros((n_perturbations, self.K))
        for i, pert_gene in enumerate(perturbation_genes):
            if pert_gene in self.gene_to_idx:
                gene_idx = self.gene_to_idx[pert_gene]
                P[i] = self.G[gene_idx]
            else:
                # 如果扰动基因不在训练集中，使用零向量
                P[i] = np.zeros(self.K)
        
        # 预测：根据公式 Y ≈ GWP^T + b
        # Y^T ≈ GWP^T，所以 Y ≈ (GWP^T)^T = P W^T G^T
        # 或者直接：Y_pred = (G @ W @ P^T)^T + b = P @ W^T @ G^T + b
        Y_pred = P @ self.W.T @ self.G.T + self.b  # (n_perturbations × n_genes)
        
        return Y_pred


def aggregate_to_pseudobulk(adata: sc.AnnData, 
                            perturbation_col: str = 'gene') -> Tuple[np.ndarray, List[str], List[str]]:
    """
    将单细胞数据聚合成伪批量（pseudobulk）
    对每个扰动条件计算平均表达
    
    Parameters:
    -----------
    adata : sc.AnnData
        单细胞数据
    perturbation_col : str
        扰动信息所在的列名
        
    Returns:
    --------
    Y : np.ndarray
        聚合后的表达矩阵 (n_perturbations × n_genes)
    perturbation_genes : List[str]
        每个扰动对应的目标基因列表
    gene_names : List[str]
        基因名称列表
    """
    # 获取扰动信息
    perturbations = adata.obs[perturbation_col].values
    
    # 获取唯一扰动（排除NC对照组）
    unique_perturbations = [p for p in np.unique(perturbations) if p != 'NC']
    
    # 聚合每个扰动的平均表达
    Y_list = []
    perturbation_genes = []
    
    for pert in unique_perturbations:
        mask = perturbations == pert
        if mask.sum() > 0:
            # 计算该扰动的平均表达
            pert_mean = np.asarray(adata[mask].X.mean(axis=0)).ravel()
            Y_list.append(pert_mean)
            perturbation_genes.append(pert)
    
    Y = np.array(Y_list)  # (n_perturbations × n_genes)
    gene_names = adata.var.index.tolist()
    
    return Y, perturbation_genes, gene_names


def compute_l2_distance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算L2距离（均方根误差）
    
    Parameters:
    -----------
    y_true : np.ndarray
        真实值 (n_samples × n_features)
    y_pred : np.ndarray
        预测值 (n_samples × n_features)
        
    Returns:
    --------
    l2_distance : float
        L2距离
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def compute_pearson_delta(y_true_delta: np.ndarray, y_pred_delta: np.ndarray) -> float:
    """
    计算Pearson Delta指标
    计算两个Delta矩阵/向量之间的Pearson相关系数
    
    Parameters:
    -----------
    y_true_delta : np.ndarray
        真实Delta矩阵 (n_perturbations × n_features) 或向量 (n_features,)
    y_pred_delta : np.ndarray
        预测Delta矩阵 (n_perturbations × n_features) 或向量 (n_features,)
        
    Returns:
    --------
    pearson_delta : float
        Pearson Delta（扰动效应的Pearson相关系数）
    """
    # 展平为一维向量
    y_true_delta_flat = y_true_delta.flatten()
    y_pred_delta_flat = y_pred_delta.flatten()
    
    # 检查是否有足够的方差
    if np.std(y_true_delta_flat) == 0 or np.std(y_pred_delta_flat) == 0:
        # 如果方差为0，返回0（表示无相关性）
        return 0.0
    
    # 计算Pearson相关系数
    try:
        corr, _ = pearsonr(y_true_delta_flat, y_pred_delta_flat)
        return corr if not np.isnan(corr) else 0.0
    except:
        return 0.0


def cross_validate(data_path: str, 
                   K: int = 10, 
                   lambda_reg: float = 0.1,
                   n_folds: int = 5,
                   random_state: int = 42,
                   use_simple_model: bool = False,
                   use_mainpy_model: bool = False,
                   use_pearson_sota_model: bool = False,
                   n_components: int = 50,
                   proportion_path: Optional[str] = None) -> Dict[str, List[float]]:
    """
    五折交叉验证
    
    Parameters:
    -----------
    data_path : str
        数据文件路径
    K : int
        PCA主成分数量（用于双线性模型）
    lambda_reg : float
        岭回归正则化系数
    n_folds : int
        交叉验证折数
    random_state : int
        随机种子
    use_simple_model : bool
        是否使用简单线性模型
    use_mainpy_model : bool
        是否使用MainPy线性模型（完全复现main.py）
    use_pearson_sota_model : bool
        是否使用PearsonSOTAModel（完全复现main_pearson_sota.py）
    n_components : int
        SVD主成分数量（用于简单模型和MainPy模型）
    proportion_path : str, optional
        程序比例CSV文件路径（用于MainPy模型和PearsonSOTAModel）
        
    Returns:
    --------
    results : Dict[str, List[float]]
        每折的评估结果
    """
    # 1. 加载数据（使用backed模式避免内存问题）
    print("Loading data...")
    adata = sc.read_h5ad(data_path, backed='r')
    print(f"Data shape: {adata.shape}")
    
    # 2. 聚合为伪批量
    print("Aggregating to pseudobulk...")
    Y, perturbation_genes, gene_names = aggregate_to_pseudobulk(adata)
    print(f"Pseudobulk shape: {Y.shape}")
    print(f"Number of perturbations: {len(perturbation_genes)}")
    
    # 保存adata的引用（简单线性模型和MainPy模型需要访问原始单细胞数据）
    if use_simple_model or use_mainpy_model or use_pearson_sota_model:
        adata_ref = adata
    else:
        adata_ref = None
    
    # 加载程序比例数据（如果使用MainPy模型或PearsonSOTAModel）
    proportion_df = None
    if (use_mainpy_model or use_pearson_sota_model) and proportion_path is not None:
        try:
            proportion_df = pd.read_csv(proportion_path)
            print(f"Loaded proportion data: {len(proportion_df)} rows")
        except Exception as e:
            print(f"Warning: Could not load proportion data: {e}")
            proportion_df = None
    
    # 3. 准备交叉验证
    n_perturbations = len(perturbation_genes)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    # 存储结果
    results = {
        'fold': [],
        'l2_distance': [],  # 原始表达的L2距离
        'l2_distance_delta': [],  # Delta的L2距离
        'pearson_delta': []  # Pearson Delta
    }
    
    # 4. 进行交叉验证
    print(f"\nStarting {n_folds}-fold cross-validation...")
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(range(n_perturbations))):
        print(f"\n=== Fold {fold_idx + 1}/{n_folds} ===")
        
        # 划分训练集和测试集
        Y_train = Y[train_idx]
        Y_test = Y[test_idx]
        train_perturbations = [perturbation_genes[i] for i in train_idx]
        test_perturbations = [perturbation_genes[i] for i in test_idx]
        
        print(f"Train perturbations: {len(train_perturbations)}")
        print(f"Test perturbations: {len(test_perturbations)}")
        
        # 训练模型
        if use_pearson_sota_model:
            # 使用PearsonSOTAModel（完全复现main_pearson_sota.py）
            model = PearsonSOTAModel(
                n_components=n_components,
                delta_alpha=10.0,
                prop_alpha=1.0
            )
            model.fit(adata_ref, train_perturbations, gene_names, proportion_df)
        elif use_mainpy_model:
            # 使用MainPy线性模型（完全复现main.py）
            model = MainPyLinearModel(
                n_components=n_components,
                cosine_alpha=10.0,
                prop_alpha=0.001,
                n_epochs=500,
                learning_rate=1e-2,
                use_torch=TORCH_AVAILABLE
            )
            model.fit(adata_ref, train_perturbations, gene_names, proportion_df)
        elif use_simple_model:
            # 使用简单线性模型
            model = SimpleLinearModel(n_components=n_components, lambda_reg=lambda_reg)
            # 需要传入原始adata以提取NC细胞，以及训练集的伪批量数据
            model.fit(adata_ref, Y_train, train_perturbations, gene_names)
        else:
            # 使用双线性模型
            model = BaselineLinearModel(K=K, lambda_reg=lambda_reg)
            model.fit(Y_train, gene_names, train_perturbations)
        
        # 预测（PearsonSOTAModel使用predict_mean方法）
        if use_pearson_sota_model:
            Y_pred = model.predict_mean(test_perturbations)
        else:
            Y_pred = model.predict(test_perturbations)
        
        # 计算扰动均值（使用训练集中所有扰动的均值作为perturbed_centroid）
        # 注意：Y_train已经是每个扰动的平均表达了，所以直接取均值即可
        perturbed_centroid = Y_train.mean(axis=0)
        
        # 计算Delta（扰动效应）
        # 真实Delta：测试集扰动相对于扰动均值的差异
        Y_test_delta = Y_test - perturbed_centroid  # (n_test_perturbations × n_genes)
        # 预测Delta：预测值相对于扰动均值的差异
        Y_pred_delta = Y_pred - perturbed_centroid  # (n_test_perturbations × n_genes)
        
        # 评估
        # L2距离：评估Delta的L2距离
        l2_dist_delta = compute_l2_distance(Y_test_delta, Y_pred_delta)
        # Pearson Delta：评估Delta之间的Pearson相关系数
        # 方法1：对所有扰动的delta一起计算Pearson（展平所有基因）
        # 这更符合竞赛评估方式：计算所有基因的delta之间的相关性
        pearson_corr = compute_pearson_delta(Y_test_delta, Y_pred_delta)
        
        # 如果结果为NaN（可能因为方差为0），尝试方法2：每个扰动单独计算
        if np.isnan(pearson_corr):
            pearson_deltas = []
            for i in range(len(Y_test_delta)):
                try:
                    pearson_delta = compute_pearson_delta(
                        Y_test_delta[i:i+1], 
                        Y_pred_delta[i:i+1]
                    )
                    if not np.isnan(pearson_delta):
                        pearson_deltas.append(pearson_delta)
                except:
                    continue
            if len(pearson_deltas) > 0:
                pearson_corr = np.mean(pearson_deltas)
            else:
                pearson_corr = 0.0  # 如果所有都是NaN，设为0
        
        # 也计算原始表达的L2距离（用于对比）
        l2_dist = compute_l2_distance(Y_test, Y_pred)
        
        print(f"L2 Distance (expression): {l2_dist:.4f}")
        print(f"L2 Distance (delta): {l2_dist_delta:.4f}")
        print(f"Pearson Delta: {pearson_corr:.4f}")
        
        # 保存结果
        results['fold'].append(fold_idx + 1)
        results['l2_distance'].append(l2_dist)
        results['l2_distance_delta'].append(l2_dist_delta)
        results['pearson_delta'].append(pearson_corr)
    
    # 5. 汇总结果
    print("\n" + "="*50)
    print("Cross-Validation Results Summary")
    print("="*50)
    print(f"Mean L2 Distance (expression): {np.mean(results['l2_distance']):.4f} ± {np.std(results['l2_distance']):.4f}")
    print(f"Mean L2 Distance (delta): {np.mean(results['l2_distance_delta']):.4f} ± {np.std(results['l2_distance_delta']):.4f}")
    print(f"Mean Pearson Delta: {np.mean(results['pearson_delta']):.4f} ± {np.std(results['pearson_delta']):.4f}")
    print("\nPer-fold results:")
    for i in range(n_folds):
        print(f"Fold {i+1}: L2(expr)={results['l2_distance'][i]:.4f}, L2(delta)={results['l2_distance_delta'][i]:.4f}, Pearson Delta={results['pearson_delta'][i]:.4f}")
    
    return results


if __name__ == "__main__":
    # 数据路径
    data_path = "data/original_data/default/obesity_challenge_1.h5ad"
    proportion_path = "data/original_data/default/program_proportion.csv"
    
    # 运行交叉验证 - PearsonSOTAModel（完全复现main_pearson_sota.py）
    print("="*60)
    print("Testing PearsonSOTA Model (Replicating main_pearson_sota.py)")
    print("="*60)
    results_pearson_sota = cross_validate(
        data_path=data_path,
        K=50,  # 不使用，但保留参数
        lambda_reg=0.1,  # 不使用，但保留参数
        n_folds=5,
        random_state=42,
        use_simple_model=False,
        use_mainpy_model=False,
        use_pearson_sota_model=True,
        n_components=50,
        proportion_path=proportion_path
    )
    
    # 保存结果
    results_df_pearson_sota = pd.DataFrame(results_pearson_sota)
    results_df_pearson_sota.to_csv("pearson_sota_cv_results.csv", index=False)
    print(f"\nPearsonSOTA Model results saved to pearson_sota_cv_results.csv")
    
    # 运行交叉验证 - MainPy线性模型（完全复现main.py）
    print("\n" + "="*60)
    print("Testing MainPy Linear Model (Replicating main.py)")
    print("="*60)
    results_mainpy = cross_validate(
        data_path=data_path,
        K=50,  # 不使用，但保留参数
        lambda_reg=0.1,  # 不使用，但保留参数
        n_folds=5,
        random_state=42,
        use_simple_model=False,
        use_mainpy_model=True,
        use_pearson_sota_model=False,
        n_components=50,
        proportion_path=proportion_path
    )
    
    # 保存结果
    results_df_mainpy = pd.DataFrame(results_mainpy)
    results_df_mainpy.to_csv("mainpy_linear_cv_results.csv", index=False)
    print(f"\nMainPy Linear Model results saved to mainpy_linear_cv_results.csv")
    
    # 运行交叉验证 - 简单线性模型（对比）
    print("\n" + "="*60)
    print("Testing Simple Linear Model (for comparison)")
    print("="*60)
    results_simple = cross_validate(
        data_path=data_path,
        K=50,  # 不使用，但保留参数
        lambda_reg=0.1,
        n_folds=5,
        random_state=42,
        use_simple_model=True,
        use_mainpy_model=False,
        n_components=50
    )
    
    # 保存结果
    results_df_simple = pd.DataFrame(results_simple)
    results_df_simple.to_csv("simple_linear_cv_results.csv", index=False)
    print(f"\nSimple Linear Model results saved to simple_linear_cv_results.csv")
    
    # 运行交叉验证 - 双线性模型（对比）
    print("\n" + "="*60)
    print("Testing Bilinear Model (for comparison)")
    print("="*60)
    results_bilinear = cross_validate(
        data_path=data_path,
        K=50,
        lambda_reg=0.1,
        n_folds=5,
        random_state=42,
        use_simple_model=False,
        use_mainpy_model=False
    )
    
    # 保存结果
    results_df_bilinear = pd.DataFrame(results_bilinear)
    results_df_bilinear.to_csv("bilinear_cv_results.csv", index=False)
    print(f"\nBilinear Model results saved to bilinear_cv_results.csv")
    
    # 汇总对比结果
    print("\n" + "="*60)
    print("Model Comparison Summary")
    print("="*60)
    print(f"PearsonSOTA Model (main_pearson_sota.py replica):")
    print(f"  Mean Pearson Delta: {np.mean(results_pearson_sota['pearson_delta']):.4f} ± {np.std(results_pearson_sota['pearson_delta']):.4f}")
    print(f"  Mean L2 Distance (delta): {np.mean(results_pearson_sota['l2_distance_delta']):.4f} ± {np.std(results_pearson_sota['l2_distance_delta']):.4f}")
    print(f"\nMainPy Model (main.py replica):")
    print(f"  Mean Pearson Delta: {np.mean(results_mainpy['pearson_delta']):.4f} ± {np.std(results_mainpy['pearson_delta']):.4f}")
    print(f"  Mean L2 Distance (delta): {np.mean(results_mainpy['l2_distance_delta']):.4f} ± {np.std(results_mainpy['l2_distance_delta']):.4f}")
    print(f"\nSimple Linear Model:")
    print(f"  Mean Pearson Delta: {np.mean(results_simple['pearson_delta']):.4f} ± {np.std(results_simple['pearson_delta']):.4f}")
    print(f"  Mean L2 Distance (delta): {np.mean(results_simple['l2_distance_delta']):.4f} ± {np.std(results_simple['l2_distance_delta']):.4f}")
    print(f"\nBilinear Model:")
    print(f"  Mean Pearson Delta: {np.mean(results_bilinear['pearson_delta']):.4f} ± {np.std(results_bilinear['pearson_delta']):.4f}")
    print(f"  Mean L2 Distance (delta): {np.mean(results_bilinear['l2_distance_delta']):.4f} ± {np.std(results_bilinear['l2_distance_delta']):.4f}")