# Baseline模型实现对比分析

## 两个实现的核心区别

### 1. **模型架构差异**

#### 我的实现 (`src/baseline_linear_model.py`)
- **模型公式**: Y ≈ GWP^T + b
- **结构**: 双线性模型（Bilinear Model）
- **矩阵维度**:
  - G: (n_genes × K) - 基因嵌入矩阵
  - W: (K × K) - 权重矩阵
  - P: (n_perturbations × K) - 扰动嵌入矩阵
  - b: (n_genes,) - 截距向量

#### main.py 的实现
- **模型公式**: delta = f(gene_embedding)
- **结构**: 简单线性映射
- **预测目标**: 扰动效应（delta = pert_mean - control_mean）
- **映射**: gene_embedding → delta (通过线性层)

---

### 2. **基因嵌入（Gene Embeddings）的构建方式**

#### 我的实现
```python
# 使用训练集中所有扰动的伪批量数据进行PCA
Y_centered = Y_train - self.b  # 中心化
self.G = self.pca.fit_transform(Y_centered.T)  # (n_genes × K)
```
- **数据源**: 所有扰动的平均表达（伪批量）
- **方法**: PCA，K=10
- **含义**: G矩阵捕捉了基因在所有扰动条件下的共表达模式

#### main.py 的实现
```python
# 使用控制组（NC）细胞进行SVD
svd = TruncatedSVD(n_components=50, random_state=42)
svd.fit(X_nc)  # X_nc是NC细胞的表达矩阵
gene_embeddings = svd.components_.T  # (n_genes, 50)
```
- **数据源**: 仅控制组（NC）细胞
- **方法**: TruncatedSVD，n_components=50
- **含义**: 基因嵌入基于正常（未扰动）状态下的表达模式

---

### 3. **扰动嵌入（Perturbation Embeddings）的构建**

#### 我的实现
```python
# 对于单基因扰动，直接取G矩阵中对应基因的行
for i, pert_gene in enumerate(perturbation_genes):
    gene_idx = self.gene_to_idx[pert_gene]
    P[i] = self.G[gene_idx]  # 取该基因的嵌入向量
```
- **逻辑**: 扰动基因X的特征 = 基因X在G矩阵中的嵌入
- **假设**: 被扰动的基因本身的特征就是扰动的最佳表征

#### main.py 的实现
```python
# 同样使用基因嵌入作为扰动特征
emb = gene_map.get(pert, default_emb)  # 获取扰动基因的嵌入
pred_delta = emb @ W.T  # 线性映射预测delta
```
- **逻辑**: 与我的实现相同，都使用基因嵌入作为扰动特征

---

### 4. **训练目标和损失函数**

#### 我的实现
```python
# 目标：最小化 ||Y - GWP^T - b||^2 + λ||W||^2
# 使用岭回归的正规方程求解
W = (G^T G + λI)^(-1) G^T (Y_train - b)^T P (P^T P + λI)^(-1)
```
- **损失函数**: L2损失 + L2正则化
- **求解方法**: 闭式解（正规方程）
- **预测目标**: 完整的表达矩阵Y

#### main.py 的实现
```python
# CosineRidge模型
class CosineRidge(nn.Module):
    def loss(self, y_pred, y_true):
        # 中心化后计算余弦相似度
        y_pred_c = y_pred - y_pred.mean(dim=1, keepdim=True)
        y_true_c = y_true - y_true.mean(dim=1, keepdim=True)
        cos_sim = cosine_similarity(y_pred_c, y_true_c, dim=1)
        cosine_loss = 1.0 - cos_sim.mean()
        l2 = torch.sum(self.W.weight ** 2)
        return cosine_loss + self.alpha * l2
```
- **损失函数**: 余弦相似度损失 + L2正则化
- **求解方法**: 梯度下降（Adam优化器，500 epochs）
- **预测目标**: 扰动效应delta（相对于控制组的差异）

---

### 5. **预测方式**

#### 我的实现
```python
# 直接预测完整的表达矩阵
Y_pred = P @ W^T @ G^T + b
```
- **输出**: 每个扰动的平均表达向量
- **用途**: 主要用于交叉验证和评估

#### main.py 的实现
```python
# 预测delta，然后加到控制组细胞上
pred_delta = emb @ W.T  # 预测扰动效应
base_cells = X_nc_all[selected_indices]  # 从NC细胞中采样
generated_cells = base_cells + pred_delta  # 加上delta
```
- **输出**: 100个单细胞表达向量（每个扰动）
- **策略**: 
  1. 预测扰动效应delta
  2. 从控制组细胞中采样基准细胞
  3. 将delta加到基准细胞上生成预测细胞
  4. 根据预测的比例采样不同细胞类型

---

### 6. **数据使用策略**

#### 我的实现
- **训练数据**: 所有扰动的伪批量（平均表达）
- **优势**: 直接学习扰动-表达映射关系
- **劣势**: 丢失了单细胞水平的异质性信息

#### main.py 的实现
- **训练数据**: 
  - 控制组细胞（用于构建基因嵌入）
  - 扰动组细胞的平均表达（用于学习delta）
- **优势**: 
  - 保留了单细胞异质性（通过采样NC细胞）
  - 预测更符合单细胞数据的分布特性
- **劣势**: 需要额外的细胞采样和组合步骤

---

### 7. **评估指标**

#### 我的实现
- **L2距离**: 预测均值与真实均值的L2距离
- **Pearson相关系数**: 相对于扰动均值的相关性（Pearson Delta）
- **评估对象**: 伪批量（平均表达）

#### main.py 的实现
- **Pearson Delta**: 相对于扰动均值的相关性
- **MMD**: 预测分布与真实分布之间的最大均值差异
- **L1距离**: 细胞类型比例的L1距离
- **评估对象**: 单细胞表达矩阵（100个细胞/扰动）

---

### 8. **关键设计差异总结**

| 方面 | 我的实现 | main.py |
|------|---------|---------|
| **模型类型** | 双线性模型 (GWP^T) | 线性映射 (gene_emb → delta) |
| **嵌入数据源** | 所有扰动的伪批量 | 仅控制组（NC）细胞 |
| **嵌入维度** | K=10 (PCA) | 50 (TruncatedSVD) |
| **损失函数** | L2 + L2正则 | 余弦相似度 + L2正则 |
| **求解方法** | 闭式解（正规方程） | 梯度下降（Adam） |
| **预测输出** | 平均表达向量 | 100个单细胞 |
| **细胞生成** | 无（仅预测均值） | 采样NC细胞 + delta |
| **评估方式** | 伪批量评估 | 单细胞分布评估 |

---

### 9. **各自的优势**

#### 我的实现
1. ✅ **理论清晰**: 严格遵循论文中的双线性模型公式
2. ✅ **计算高效**: 闭式解，无需迭代训练
3. ✅ **参数少**: 仅需学习W矩阵（K×K）
4. ✅ **适合快速baseline**: 实现简单，易于理解

#### main.py 的实现
1. ✅ **更符合竞赛要求**: 生成单细胞数据，支持MMD评估
2. ✅ **保留异质性**: 通过采样NC细胞保留单细胞变异
3. ✅ **损失函数更优**: 余弦相似度损失对齐Pearson Delta指标
4. ✅ **实用性强**: 完整的训练-推理流程，可直接提交

---

### 10. **建议改进方向**

#### 对我的实现的改进
1. 添加单细胞生成逻辑（采样NC细胞 + delta）
2. 使用余弦相似度损失替代L2损失
3. 考虑使用控制组细胞构建基因嵌入（可能更稳定）
4. 增加细胞类型比例预测

#### 对main.py的改进
1. 可以尝试双线性模型结构（可能表达能力更强）
2. 考虑使用所有扰动数据构建基因嵌入（而不仅仅是NC）
3. 可以尝试更小的嵌入维度（如K=10）以减少过拟合风险

---

## 总结

两个实现的核心区别在于：
- **我的实现**: 严格遵循论文的双线性模型，预测伪批量表达
- **main.py**: 实用的线性映射模型，预测扰动效应并生成单细胞数据

main.py的实现更符合竞赛的实际需求（需要生成单细胞数据），而我的实现更符合论文的理论框架。两者可以结合使用，取长补短。
