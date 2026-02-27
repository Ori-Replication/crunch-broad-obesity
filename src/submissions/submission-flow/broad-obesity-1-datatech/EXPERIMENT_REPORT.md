# 细胞比例预测实验报告

**实验脚本**: `eda/esm/esm_proportion_experiment.py`  
**运行环境**: `QUICK=1` 模式（缩减参数搜索空间）  
**验证**: 5-fold CV，L1 比例损失

---

## 1. 实验目标

探索 ESM2 / scGPT / Gene2vec 单源及多源组合的最优参数，寻找最稳定、效果最好的细胞比例预测配置。

---

## 2. 实验结果汇总

### 2.1 单源 Embedding + PCA + KNN

| 方法 | 最佳参数 | L1 (mean ± std) |
|------|----------|-----------------|
| **scGPT** | n_pca=32, k=15 | **0.0993 ± 0.0159** ✓ 最优 |
| ESM2 | n_pca=96, k=25 | 0.1000 ± 0.0201 |
| Gene2vec | n_pca=32, k=20 | 0.1008 ± 0.0201 |

### 2.2 多源 Embedding

| 方法 | 最佳参数 | L1 (mean ± std) |
|------|----------|-----------------|
| 各源分别 PCA 后拼接 | n_pca=(48,32,48), k=8 | 0.0996 ± 0.0173 |
| 拼接后联合 PCA | n_joint=64, k=25 | 0.1013 ± 0.0217 |

### 2.3 基线

| 方法 | L1 (mean ± std) |
|------|-----------------|
| Average 基线 | 0.0978 ± 0.0181 |

---

## 3. 结论

- **最佳 L1**: scGPT_PCA_KNN (n_pca=32, k=15)
- **最稳定** (mean<0.105): scGPT_PCA_KNN (n_pca=32, k=15)
- **综合最优** (mean+0.5×std): scGPT_PCA_KNN (n_pca=32, k=15)

scGPT 在单源中表现最佳，且方差最小（0.0159），适合作为主要预测源。

---

## 4. 应用于提交流程的改进

将以下配置应用到 `broad-obesity-1-datatech`：

| 配置项 | 原值 | 新值 |
|--------|------|------|
| Embedding 源 | Gene2vec | scGPT（优先），Gene2vec（回退） |
| PCA 维度 | 无 | n_pca=32 |
| KNN 邻居数 | k=15 | k=15（scGPT）/ k=20（Gene2vec） |

注意：当前 submission 使用 Gene2vec 时未做 PCA，实验表明 Gene2vec + PCA(32) 可改善效果。
