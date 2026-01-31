# Baseline Linear Model 实现

## 概述

本目录包含基线线性模型的实现，基于论文中描述的模型：Y ≈ GWP^T + b

## 文件说明

- `src/baseline_linear_model.py`: 模型实现和交叉验证代码
- `baseline_cv_results.csv`: 五折交叉验证结果
- `experiment_results.md`: 详细的实验结果和分析

## 使用方法

### 运行交叉验证

```bash
python src/baseline_linear_model.py
```

### 代码结构

#### 主要类和方法

1. **BaselineLinearModel**: 基线线性模型类
   - `fit()`: 训练模型
   - `predict()`: 预测扰动效果

2. **辅助函数**:
   - `aggregate_to_pseudobulk()`: 将单细胞数据聚合成伪批量
   - `compute_l2_distance()`: 计算L2距离
   - `compute_pearson_correlation()`: 计算Pearson相关系数
   - `cross_validate()`: 执行交叉验证

### 参数说明

- `K`: PCA主成分数量，默认10
- `lambda_reg`: 岭回归正则化系数，默认0.1
- `n_folds`: 交叉验证折数，默认5
- `random_state`: 随机种子，默认42

## 实验结果

详细结果请参见 `experiment_results.md`

### 快速总结

- **平均L2距离**: 0.0542 ± 0.0041
- **平均Pearson相关系数**: -0.1116 ± 0.3397

## 注意事项

1. 数据路径: 代码中硬编码了数据路径 `data/original_data/default/obesity_challenge_1.h5ad`，如需修改请编辑代码
2. 内存使用: 数据较大，确保有足够内存
3. 运行时间: 完整交叉验证大约需要几分钟

## 依赖库

- numpy
- pandas
- scanpy
- scikit-learn
- scipy

## 参考

- Baseline模型描述: `docs/baseline.md`
- 数据说明: `docs/数据说明.md`
- 赛题说明: `docs/赛题.md`
