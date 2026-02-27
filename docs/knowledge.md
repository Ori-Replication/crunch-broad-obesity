当前项目为一个单细胞基因扰动效应预测的数据科学竞赛项目。
数据集在 data/ 下，original_data/default 是赛题数据。
如果你需要了解竞赛详情，请查看 docs/赛题.md
如果需要了解竞赛提供的原始数据结构，请查看 docs/数据说明.md
如果代码数据探索性数据分析，请放在 eda/ 下
eda/ 下不同子文件夹代表了之前进行的不同探索，部分形成了总结文档。
eda/xatlas: 关于外部全基因组扰动数据集 xatlas 的探索
eda/deg: 差异表达基因的探索
eda/perturbmean: 使用简单 perturbmean 方法进行交叉验证的结果。PerturbMean 虽然 CV 很高，但在线上测试中发生了过拟合。我们一般认为，如果向量与 PerturbMean 余弦相似度高，则很有可能发生过拟合。
eda/regression: 基于 regression-baseline 的回归方法探索，结合 xatlas 迁移方法
eda/flow: 基于 PCA 的流匹配（Flow Matching）生成模型探索，PM+Flow 组合取得 +3.4% 提升
eda/flow_xatlas: 流匹配 + xatlas 条件，PM+FlowXA 优于 PCA_HEK 但略逊于纯 PM+Flow
eda/gene2vec: Gene2vec 预训练 embedding 与竞赛基因重合度分析，97.1% 预测目标有 embedding
eda/scgpt: scGPT Embedding Layer 下载与重合度分析，100% 预测目标有 embedding，优于 Gene2vec
eda/gene2vec_flow: Gene2vec+Flow 条件扩展 + Ridge 细胞比例预测探索
eda/MMD: MMD 采样策略探索——除均值外，如何采样以更好地近似整体协方差
eda/GSE217812: 脂肪细胞 CROP-Seq 数据集成，FlowXA+GSE217812 暂时在 Pearson 上取得最佳效果.
eda/genept: GenePT text embedding 覆盖度分析，FlowXA+GenePT_gpc20 在 CV 上取得 +11.2% Pearson 提升.
eda/GenePT: GenePT 文本 embedding（NCBI 描述 + ada-002）重合度分析，99.5% 预测目标覆盖；细胞比例预测略逊于 Average 基线，优于 Gene2vec.
如果代码属于临时性测试脚本，请放在 tmp/ 下
如果代码属于正式竞赛方案的一部分，请放在 src/ 下
src/submission: 正式的提交代码
data/prism: 一个外部数据集，包含多个人类的基因扰动数据集。
data/xatlas: 一个外部数据集，包含两个覆盖面较广的细胞系扰动数据，几乎覆盖了所有的本竞赛中的扰动基因。
工作环境应当为 broad，通过 conda activate broad 激活。你可以安装所需的库。
如需绘图，绘图时，图片上出现的文字都应当是英文。
