# Perturbmean Baseline 实现说明

## 方法概述

本实现使用简单的 Perturbmean baseline 方法：
- **训练阶段**：计算所有训练扰动的平均 Delta（相对于 NC 控制组）
- **推理阶段**：从 NC 中采样 100 个细胞，加上平均 Delta 得到预测

## 主要修改

### 1. `train.py` - 训练脚本

**功能**：
- 计算所有扰动的平均 Delta（相对于 NC 控制组）
- 计算所有扰动的平均细胞比例
- 保存必要的模型数据

**输出文件**：
- `average_delta.pkl`: 平均 Delta 向量 (21592,)
- `average_proportions.pkl`: 平均细胞比例 [pre_adipo, adipo, lipo, other]
- `control_mean.pkl`: NC 控制组的平均表达谱
- `nc_indices.pkl`: NC 细胞的索引

### 2. `infer.py` - 推理脚本

**功能**：
- 从 NC 中采样 100 个细胞（保证采样后均值接近整体均值）
- 对每个扰动，使用采样细胞 + 平均 Delta 生成预测
- 使用平均细胞比例（已满足约束条件）

**关键函数**：
- `sample_nc_cells_with_mean_match()`: 从 NC 中采样，使得采样后的均值尽可能接近整体均值
  - 使用多次随机采样（默认 500 次），选择均值最接近的样本
  - 保证采样后的均值与整体均值的差异最小

**约束条件**：
- ✅ `lipo <= adipo`：Lipogenic 是 Adipocyte 的子集
- ✅ `pre_adipo + adipo + other = 1`：三个主要类型的比例和为 1

### 3. `main.py` - 主执行脚本

**修改**：
- 更新数据路径为绝对路径，使用 `os.path.join(BASE_DIR, "data")`

## 数据路径

所有数据文件位于：
```
/home/mutianhong/work/crunch-broad/CrunchDAO-obesity/src/submissions/submissions-egg/broad-obesity-1-datatech/data/
```

## 运行方法

```bash
cd /home/mutianhong/work/crunch-broad/CrunchDAO-obesity/src/submissions/submissions-egg/broad-obesity-1-datatech
conda activate broad
python main.py
```

## 输出文件

### 训练输出 (`resources/`)
- `average_delta.pkl`: 平均 Delta
- `average_proportions.pkl`: 平均比例
- `control_mean.pkl`: 控制组均值
- `nc_indices.pkl`: NC 细胞索引

### 推理输出 (`prediction/`)
- `prediction.h5ad`: 预测的基因表达谱
- `predict_program_proportion.csv`: 预测的细胞比例

## 方法特点

1. **简单高效**：直接使用训练集的平均 Delta，无需复杂模型
2. **均值匹配采样**：保证采样后的 NC 细胞均值接近整体均值
3. **满足约束**：细胞比例自动满足赛题要求
4. **内存友好**：使用 HDF5 临时文件避免内存溢出

## 实验结果

在本地测试集上的表现：
- 使用训练集中所有 122 个扰动的平均 Delta
- 每个扰动生成 100 个细胞
- 细胞比例使用训练集的平均比例（已归一化）

## 注意事项

1. **采样策略**：每个扰动都会重新采样 100 个 NC 细胞，保证采样的多样性
2. **均值匹配**：采样时会尝试多次（默认 500 次），选择均值最接近的样本
3. **比例约束**：训练时已确保比例满足约束条件，推理时直接使用
