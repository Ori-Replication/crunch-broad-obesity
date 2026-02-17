# 改进思路：外部资料与 Gene Embedding 方向

> 基于 PM+FlowImp_ode100_pc5_a0.11（Pearson 0.2301），探索小模型 + 外部 Embedding 四两拨千斤的路线。

## 一、当前方法的瓶颈

1. **条件向量来源单一**：Flow 的 condition `c` 仅来自竞赛数据的 PCA gene loadings（基因在「扰动 × 基因」空间中的载荷），未见基因或低表达基因可能 fallback 到纯 PerturbMean。

2. **xatlas 的局限**：flow_xatlas 已证明 HEK293T 迁移信号弱（细胞类型差异大），PM+FlowXA 略逊于 PM+Flow。

3. **流形提炼能力**：当前仅用 5 维 PCA，可考虑引入更丰富的基因关系表征，帮助模型理解「相似基因 → 相似扰动效应」。

---

## 二、推荐外部资料与 Embedding 资源

### 1. 基因 Embedding（无需运行大模型，直接查表）

| 资源 | 说明 | 获取方式 | 用途 |
|------|------|----------|------|
| **Gene2vec** | 基于 GEO 共表达的 200 维基因向量，可表征基因功能相似性 | [GitHub: jingcheng-du/Gene2vec](https://github.com/jingcheng-du/Gene2vec)，`pre_trained_emb/` | 对未见基因：用 embedding 相似度做 k-NN 插值，或 `c = [gene_loadings \| gene2vec]` 扩展条件 |
| **scGPT GeneEmbedding** | scGPT 的 `GeneEmbedding` 类，支持基因网络推断 | scGPT 的 `scgpt.tasks.grn` 模块，可从预训练模型提取 | 作为条件扩展，需安装 scGPT 并加载模型 |
| **Gene Ontology (GO) 语义相似度** | 基于 GO 术语的基因功能相似度矩阵 | [goatools](https://github.com/tanghaibao/goatools) 或 [Semantic similarity API](http://geneontology.org/docs/go-enrichment-analysis/) | 构造基因-基因相似度，用于 k-NN 插值 |

**实现建议**：
- 将 Gene2vec 嵌入与 `gene_loadings` 拼接：`c = [gene_loadings, gene2vec_gene]`，需对齐维度（PCA 降维或线性映射到相同尺度）。
- 对 fallback 基因：在 Gene2vec 空间找 k 近邻训练基因，用加权平均的 `gene_loadings` 作为 proxy condition。

### 2. 基因关系网络（无需大模型）

| 资源 | 说明 | 获取方式 | 用途 |
|------|------|----------|------|
| **STRING** | 蛋白-蛋白互作，综合置信度分数 | [string-db.org](https://string-db.org/cgi/download.pl)，`protein.links.v12.0.txt` 或 API | 对扰动基因 G，找其 STRING 邻居中在训练集的基因，加权平均其 delta 或 loadings |
| **TRRUST / ChIP-Atlas** | TF → 靶基因调控关系 | TRRUST v2、ChIP-Atlas | 竞赛多为 TF 扰动，可构造「TF 调控网络」特征 |
| **KEGG / Reactome 通路** | 通路-基因 membership | KEGG API、Reactome | 同通路基因可能有相似扰动模式，作为辅助条件 |

**实现建议**：
- 对未见基因 G：  
  `c_proxy = Σ w_i * loadings[train_gene_i]`，其中 `w_i` 为 G 与 train_gene_i 的 STRING 分数或 GO 相似度。

### 3. 生物学先验（flow_xatlas 已提及）

> 实验总结：「用生物学先验（通路、网络）构造更稳健的条件特征」

- **通路富集**：对每个扰动基因，提取其参与的 KEGG/Reactome 通路，用 one-hot 或通路 embedding 作为条件的一部分。
- **TF 身份**：150/157 为 TF，可加 TF 家族、DNA 结合域等离散特征，embedding 后拼入条件。

### 4. 外部扰动数据（谨慎使用）

| 资源 | 说明 | 注意 |
|------|------|------|
| **Prism**（项目已有） | 多个人类基因扰动数据集 | 检查是否含脂肪/代谢相关细胞系，比 xatlas 更贴近任务 |
| **xatlas/Orion** (2025.06) | Xaira 全基因组 Perturb-seq，HCT116/HEK293T | 已试，迁移有限；可选其他 cell line 的子集 |
| **Norman et al. 单细胞扰动** | 部分与 xatlas 重叠 | 需筛选与脂肪分化相关的实验 |

**建议**：优先用 Prism 中与脂肪/前脂肪相近的细胞类型；若仍无增益，则转向 Embedding/网络先验。

---

## 三、具体改进路线（小模型优先）

### 路线 A：扩展 Flow 的 condition（推荐优先）

1. **Gene2vec 扩展**：
   - 下载 Gene2vec 预训练 embedding。
   - 对 21592 基因做名称映射（需处理基因 ID 转换，如 symbol → Entrez）。
   - `c = [gene_loadings; PCA(gene2vec)[:d]]`，d=5~20，总维度 10~25。
   - 对 embedding 缺失的基因：k-NN 插值或零填充。

2. **STRING k-NN 插值（解决 fallback）**：
   - 对 `gene_name not in gene_loadings_df.index` 的基因，用 STRING 找 k 个训练基因邻居。
   - `c = weighted_avg(neighbors' loadings)`，权重=STRING 分数。
   - 这样 fallback 基因也能获得非零的 flow 贡献。

### 路线 B：多源 condition 融合

- `c = α·gene_loadings + β·gene2vec_pca + γ·pathway_features`
- 用轻量回归或小型 MLP 学习 `[loadings, embedding, pathway] → delta`，保持总参数量在几百到几千。

### 路线 C：Ensemble 与后处理

- 在现有 PM+Flow 基础上，增加一条「PM+Flow_gene2vec」分支，与 PM+Flow 做加权平均或 stacking。
- 继续做 Winsorize、clip 等后处理，控制极端预测。

---

## 四、实施优先级建议

| 优先级 | 方向 | 工作量 | 预期收益 |
|--------|------|--------|----------|
| 1 | Gene2vec 条件扩展 | 中 | 对未见/低表达基因可能提升 |
| 2 | STRING k-NN 插值（fallback） | 低 | 直接减少纯 PM fallback 比例 |
| 3 | Prism 脂肪相关数据探索 | 中 | 若有合适细胞系，可弥补 xatlas 不足 |
| 4 | GO 相似度 / 通路特征 | 中 | 条件更稳健，可能小幅提升 |
| 5 | scGPT GeneEmbedding | 高 | 依赖 scGPT 环境，收益不确定 |

---

## 五、参考链接

- Gene2vec: https://github.com/jingcheng-du/Gene2vec  
- STRING: https://string-db.org/cgi/download.pl  
- scGPT: https://github.com/bowang-lab/scGPT（GeneEmbedding 在 `scgpt.tasks.grn`）  
- flow_xatlas 建议：`eda/flow_xatlas/实验总结.md` —「用生物学先验（通路、网络）构造更稳健的条件特征」
