# 人工智能期末大作业执行蓝图

## 1. 作业目标与提交要求

本项目目标是完成一篇基于监督学习的分类应用实验报告，覆盖从数据采集、数据清洗、特征工程、模型训练、模型评估到论文撰写与 PDF 提交的完整流程。最终提交物为一份排版规范、正文不少于 4000 字的 PDF 课程论文。

硬性要求如下：

- 论文主题：自选一个分类问题，结合监督学习技术完成完整实验报告。
- 正文字数：不少于 4000 字。
- 提交格式：PDF 电子版。
- 提交邮箱：1191618741@qq.com。
- 邮件标题：AI+姓名+学号。
- 截止时间：2025 年 6 月 15 日 23:59。
- 论文结构建议：引言 → 综述 → 深入分析 → 应用实现 → 实验 → 总结。
- 核心原则：复杂度不重要，分析深度才重要。

评分权重：

- 技术分析与应用设计：40%。
- 实验完整性与代码质量：30%。
- 写作规范与排版：30%。

加分方向：

- 从零实现算法核心逻辑。
- 自行采集或构造数据集。
- 设计创新性应用场景。
- 进行多模型系统对比。
- 做深入的失败案例分析。

本项目采用“外部公开数据集 + 手动采集 + 可复现实验代码 + Overleaf 编译论文”的路线，重点保证流程完整、分析扎实、图表充分、结论可解释。

## 2. 项目选题与数据来源

### 2.1 选题

论文选题确定为：**基于 UCI Adult 数据集的个人收入水平二分类预测研究**。

分类任务定义：根据人口统计学与工作相关特征，预测一个人的年收入是否超过 50,000 美元。目标变量为：

- `<=50K`：年收入不超过 50K。
- `>50K`：年收入超过 50K。

该任务适合作为课程大作业主线，原因如下：

- 属于标准监督学习二分类问题，符合课程要求。
- 数据来源权威，便于引用和复现。
- 同时包含数值特征和类别特征，能体现特征编码、标准化和预处理流程。
- 数据中存在 `?` 缺失标记，能体现数据清洗过程。
- 类别分布不完全均衡，适合同时报告 Accuracy、Macro-F1、Micro-F1。
- 样本量适中，CPU 环境下可以完成 MLP 训练和超参数实验。

### 2.2 数据来源

数据集采用 UCI Machine Learning Repository 中的 Adult/Census Income 数据集。

- 官方数据集页面：https://archive.ics.uci.edu/ml/datasets/Adult
- 原始数据目录：https://archive.ics.uci.edu/ml/machine-learning-databases/adult/
- 数据说明文件：https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names
- 推荐引用：Becker, B. & Kohavi, R. (1996). Adult [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5XW20

数据集基本信息：

- 样本数：48,842。
- 特征数：14。
- 任务类型：二分类。
- 特征类型：类别型与整数型混合。
- 缺失值：存在，原始文件中用 `?` 表示。
- 原始划分：`adult.data` 与 `adult.test`，但本项目会合并后重新按课程要求划分为 60%/20%/20%。

### 2.3 手动采集步骤

后续执行时手动完成数据采集，并在论文中描述“手动从 UCI 官方页面下载原始文件”的过程。

建议目录：

```text
Finals/
├── data/
│   └── raw/
│       ├── adult.data
│       ├── adult.test
│       └── adult.names
├── data/
│   └── processed/
├── figures/
├── results/
├── src/
└── paper/
```

采集步骤：

1. 打开 UCI Adult 官方数据目录。
2. 下载 `adult.data`、`adult.test`、`adult.names`。
3. 将三个文件放入 `data/raw/`。
4. 在报告中说明数据来源、下载时间、数据集用途与 DOI。
5. 代码读取本地文件，不依赖 sklearn 内置数据集，体现外部数据提取过程。

## 3. 数据处理与特征工程方案

### 3.1 字段设计

Adult 数据集字段如下：

```text
age, workclass, fnlwgt, education, education-num, marital-status,
occupation, relationship, race, sex, capital-gain, capital-loss,
hours-per-week, native-country, income
```

数值特征：

- `age`
- `fnlwgt`
- `education-num`
- `capital-gain`
- `capital-loss`
- `hours-per-week`

类别特征：

- `workclass`
- `education`
- `marital-status`
- `occupation`
- `relationship`
- `race`
- `sex`
- `native-country`

标签：

- `income`

### 3.2 清洗流程

代码应完成以下清洗逻辑：

1. 分别读取 `adult.data` 和 `adult.test`。
2. 跳过 `adult.test` 文件中的说明行。
3. 设置统一列名。
4. 去除字符串字段首尾空格。
5. 将 `adult.test` 中标签末尾的句点去掉，例如 `>50K.` 转为 `>50K`。
6. 将 `?` 统一识别为缺失值。
7. 统计每列缺失值数量，并输出缺失值统计表。
8. 删除含缺失值的样本，或在论文中说明选择删除的理由。
9. 检查重复样本，可统计但不强制删除，除非对结果影响明显。
10. 将标签编码为二值：`<=50K -> 0`，`>50K -> 1`。

缺失值处理策略建议：删除含缺失值样本。理由是 Adult 官方说明中去除 unknown 后仍有 45,222 条样本，样本量仍然充足，直接删除可以避免类别型缺失插补带来的额外假设，便于论文解释。

### 3.3 数据探索要求

必须生成并保存以下探索性分析结果：

- 缺失值统计表：每个字段的缺失数量和缺失比例。
- 类别分布图：`income` 标签的柱状图。
- 数值特征直方图：至少展示 `age`、`education-num`、`hours-per-week`。
- 类别特征分布图：至少展示 `workclass`、`education`、`occupation` 的 Top-N 频数。
- 类别不均衡说明：计算 `<=50K` 与 `>50K` 的比例。

图表保存到 `figures/`，命名建议：

```text
figures/missing_values.png
figures/label_distribution.png
figures/numeric_feature_histograms.png
figures/categorical_feature_distribution.png
```

### 3.4 数据划分

必须严格按课程要求执行：

- 训练集：60%。
- 验证集：20%。
- 测试集：20%。
- 使用 `stratify=y` 保证分层抽样。
- 固定 `random_state=42`。
- 先划分数据集，再进行任何需要 `fit` 的预处理。

推荐实现方式：

1. 第一次 `train_test_split`：划分出 60% 训练集和 40% 临时集。
2. 第二次 `train_test_split`：将临时集等分为 20% 验证集和 20% 测试集。
3. 两次划分均使用 `stratify`。

### 3.5 预处理原则

预处理必须避免数据泄漏：

- `StandardScaler` 只在训练集数值特征上 `fit`。
- 验证集和测试集只使用训练集拟合得到的 scaler 进行 `transform`。
- `OneHotEncoder` 只在训练集类别特征上 `fit`。
- 验证集和测试集只进行 `transform`。
- `OneHotEncoder` 设置 `handle_unknown="ignore"`，避免验证/测试集中出现未见类别时报错。

推荐使用 `ColumnTransformer` 或显式分别处理数值与类别特征。

## 4. 模型方案与实验设计

### 4.1 主模型：MLP

论文技术评价和实验主模型统一采用 MLP。推荐网络结构：

```text
Input -> Linear(input_dim, 128) -> ReLU -> Dropout(0.3)
      -> Linear(128, 64) -> ReLU -> Dropout(0.3)
      -> Linear(64, 2)
```

推荐训练配置：

- 损失函数：`CrossEntropyLoss`。
- 优化器：`AdamW`。
- 初始学习率：`0.001`。
- 权重衰减：`1e-4`。
- Batch size：`64`。
- Epoch：可先设为 `80` 或 `100`，避免 CPU 训练时间过长；如训练很快可扩展到 `200`。
- 学习率调度器：可使用 `CosineAnnealingLR`，也可在基础版本中不使用，但论文中要如实说明。
- 随机种子：`42`。
- 设备：CPU。

训练过程需要记录：

- 每个 epoch 的训练 Loss。
- 每个 epoch 的验证 Loss。
- 每个 epoch 的训练 Accuracy。
- 每个 epoch 的验证 Accuracy。
- 最优验证集指标对应的模型参数。

### 4.2 Baseline 与对比模型

为了增强结果分析说服力，保留三类对比：

1. 多数类 baseline：所有样本都预测为训练集中最多的类别。
2. 随机 baseline：按训练集类别先验概率随机预测，或均匀随机预测。
3. Logistic Regression：作为传统线性分类器强 baseline。

论文中需要解释：

- 多数类 baseline 用于衡量类别不均衡下的最低有效参考。
- 随机 baseline 用于衡量模型是否真正学习到数据规律。
- Logistic Regression 用于判断非线性 MLP 是否相对线性模型有收益。

### 4.3 评估指标

由于 Adult 是二分类且类别不完全均衡，建议同时报告：

- Accuracy。
- Precision。
- Recall。
- F1-score。
- Macro-F1。
- Micro-F1。
- Confusion Matrix。

主指标建议使用 Macro-F1 与 Accuracy 双指标：

- Accuracy 反映整体预测正确率。
- Macro-F1 能避免多数类占比过高导致结果被掩盖。

### 4.4 超参数敏感性分析

至少分析 2 个超参数，使用控制变量法：每次只改变一个参数，其余参数固定。

必做参数：

1. 学习率 `lr`
   - 候选值：`0.1`、`0.01`、`0.001`、`0.0001`。
   - 观察点：过大是否震荡，过小是否收敛慢，哪个值验证集 Macro-F1 最优。

2. Dropout
   - 候选值：`0.1`、`0.3`、`0.5`。
   - 观察点：正则化强度过弱是否过拟合，过强是否欠拟合。

可选扩展参数：

3. 隐层结构
   - 候选值：`(64,)`、`(128,64)`、`(256,128,64)`。

4. Batch size
   - 候选值：`16`、`32`、`64`、`128`。

每组超参数实验需要输出：

- 参数取值。
- 验证集 Accuracy。
- 验证集 Macro-F1。
- 测试集 Accuracy。
- 测试集 Macro-F1。
- 最优参数标记。

图表保存建议：

```text
figures/hparam_lr.png
figures/hparam_dropout.png
results/hparam_results.csv
```

## 5. 代码产物规划

### 5.1 推荐目录结构

```text
Finals/
├── final.md
├── overview.md
├── data/
│   ├── raw/
│   └── processed/
├── figures/
├── results/
├── models/
├── src/
│   ├── prepare_data.py
│   ├── train_mlp.py
│   ├── train_baselines.py
│   ├── analyze_hparams.py
│   └── plot_results.py
└── paper/
    ├── main.tex
    └── references.bib
```

### 5.2 文件职责

`src/prepare_data.py`：

- 读取 `data/raw/adult.data` 与 `data/raw/adult.test`。
- 完成字段命名、空格清理、标签统一、缺失值处理。
- 输出数据探索统计。
- 生成数据探索图表。
- 完成 60/20/20 分层划分。
- 拟合训练集预处理器，转换 train/val/test。
- 保存处理后的数据或供训练脚本调用。

`src/train_mlp.py`：

- 加载预处理后的数据。
- 构建 MLP 模型。
- 训练模型并记录 train/val 曲线。
- 保存最佳模型到 `models/mlp_best.pt`。
- 保存训练日志到 `results/mlp_training_log.csv`。
- 在测试集上输出最终指标。

`src/train_baselines.py`：

- 训练 Logistic Regression。
- 计算多数类 baseline。
- 计算随机 baseline。
- 保存对比结果到 `results/baseline_results.csv`。

`src/analyze_hparams.py`：

- 使用控制变量法运行学习率和 Dropout 实验。
- 保存结果到 `results/hparam_results.csv`。
- 生成超参数敏感性折线图。

`src/plot_results.py`：

- 读取训练日志、评估结果和混淆矩阵数据。
- 统一生成论文需要的最终图表。

### 5.3 编码与运行注意事项

当前环境中 PowerShell 对中文脚本文本和中文文件内容存在编码破坏风险。后续创建中文 Markdown 或 `.tex` 文件时，优先使用 Python 以 UTF-8 写入；如果 Python 脚本需要包含大量中文正文，必须确认脚本输入链路本身不会先被 PowerShell 转码为问号。

推荐写入方式：

```python
from pathlib import Path
Path("overview.md").write_text(content, encoding="utf-8")
```

不要用 PowerShell 的 `Set-Content` 或 here-string 直接写入大量中文内容。

## 6. 论文 `.tex` 写作规划

### 6.1 编译方式

本机当前不假设具备 LaTeX 环境。后续只需生成 `.tex`、图片、表格和参考文献文件，最终上传 Overleaf 编译。

Overleaf 建议配置：

- 编译器：XeLaTeX。
- 中文支持：`ctexart` 或 `ctexrep`。
- 图片路径：引用 `figures/`。
- 参考文献：使用 `references.bib` 或手写参考文献列表。

### 6.2 论文结构

建议 `paper/main.tex` 采用以下章节：

1. 引言
   - 监督学习分类任务背景。
   - 收入预测问题的现实意义。
   - 本文工作概述：数据清洗、MLP 建模、baseline 对比、超参分析。

2. 监督学习技术综述
   - 逻辑回归。
   - 决策树与随机森林。
   - 支持向量机。
   - 朴素贝叶斯。
   - K 近邻。
   - 神经网络与 MLP。
   - 横向对比表：模型假设、优点、缺点、适用场景。

3. MLP 技术深入分析
   - 感知机与多层前馈网络结构。
   - 线性变换、激活函数、隐藏层表达能力。
   - Softmax 与交叉熵损失。
   - 反向传播与梯度下降基本流程。
   - Dropout 与权重衰减的正则化作用。
   - MLP 优缺点与个人理解。

4. 应用设计与数据处理
   - UCI Adult 数据集介绍。
   - 字段说明。
   - 数据采集方式。
   - 缺失值检测与处理。
   - 类别分布与特征分布分析。
   - 数据划分和防止数据泄漏的预处理流程。

5. 实验设计
   - 运行环境：Windows、Python 3.11.6、PyTorch 2.11.0+cpu、scikit-learn 1.8.0、numpy 2.4.5、pandas 3.0.0、matplotlib 3.10.8、tqdm 4.67.1。
   - MLP 网络结构和训练参数。
   - Baseline 设计。
   - 评价指标。
   - 超参数敏感性分析设计。

6. 实验结果与分析
   - 训练曲线分析：是否收敛、是否过拟合。
   - 主模型与 baseline 对比。
   - 混淆矩阵错误分析。
   - 学习率敏感性分析。
   - Dropout 敏感性分析。
   - 失败案例或局限性分析。

7. 总结与反思
   - 总结本文完成的工作。
   - 说明 MLP 在表格数据上的表现与限制。
   - 反思数据特征、类别不均衡、模型复杂度和泛化能力。
   - 给出后续改进方向：更多模型对比、特征选择、公平性分析、集成学习等。

### 6.3 必备图表清单

论文中至少包含以下图表：

- 表 1：监督学习方法横向对比表。
- 表 2：Adult 数据集字段说明。
- 表 3：缺失值统计表。
- 表 4：主模型与 baseline 指标对比表。
- 表 5：学习率敏感性分析表。
- 表 6：Dropout 敏感性分析表。
- 图 1：标签类别分布图。
- 图 2：数值特征分布直方图。
- 图 3：训练 Loss 与 Accuracy 曲线。
- 图 4：测试集混淆矩阵。
- 图 5：学习率敏感性折线图。
- 图 6：Dropout 敏感性折线图。

## 7. 实验分析写作重点

### 7.1 超参数影响分析

学习率分析要回答：

- 学习率过大时，训练是否不稳定或验证指标震荡。
- 学习率过小时，模型是否收敛缓慢或欠拟合。
- 最优学习率为什么在验证集上表现更好。

Dropout 分析要回答：

- Dropout 较小时，是否训练集表现更好但验证集提升有限。
- Dropout 较大时，是否导致模型表达能力不足。
- 最优 Dropout 是否在抑制过拟合和保持表达能力之间取得平衡。

### 7.2 混淆矩阵错误分析

错误分析要重点讨论：

- `>50K` 类别是否更容易被误判为 `<=50K`。
- 类别不均衡是否导致模型偏向多数类。
- 某些职业、教育水平或工作时长是否与收入高度相关。
- 表格特征是否缺少地区经济水平、行业细分等影响收入的重要信息。

### 7.3 过拟合判断

通过训练曲线判断：

- 如果训练 Loss 持续下降而验证 Loss 上升，说明存在过拟合。
- 如果训练和验证 Accuracy 都较低，说明可能欠拟合。
- 如果训练和验证曲线同步收敛且差距较小，说明泛化较稳定。

### 7.4 Baseline 对比分析

与 baseline 对比时要说明：

- 相比多数类 baseline，模型是否真正超过“只猜多数类”的水平。
- 相比随机 baseline，模型是否学习到了有效特征模式。
- 相比 Logistic Regression，MLP 是否带来非线性建模收益。
- 如果 MLP 优势不明显，应从表格数据特征、样本规模、特征工程不足等角度反思，而不是只报告分数。

## 8. 后续执行顺序

推荐按以下顺序推进：

1. 创建目录结构：`data/raw`、`data/processed`、`figures`、`results`、`models`、`src`、`paper`。
2. 手动下载 UCI Adult 原始数据到 `data/raw/`。
3. 编写并运行 `prepare_data.py`，完成清洗、探索、划分和预处理。
4. 编写并运行 `train_baselines.py`，得到 baseline 指标。
5. 编写并运行 `train_mlp.py`，得到主模型结果和训练曲线。
6. 编写并运行 `analyze_hparams.py`，完成至少两个超参数敏感性实验。
7. 编写或运行 `plot_results.py`，统一生成论文图表。
8. 编写 `paper/main.tex` 和 `paper/references.bib`。
9. 上传 `paper/`、`figures/`、必要表格到 Overleaf。
10. 使用 XeLaTeX 编译生成 PDF。
11. 检查 PDF 字数、图表、引用、个人信息、提交邮箱和邮件标题。

## 9. 验收标准

### 9.1 数据与代码验收

- 能从 `data/raw/` 中读取 UCI Adult 原始文件。
- 能输出清洗前后样本数。
- 能输出缺失值统计和类别比例。
- 能完成 60/20/20 分层划分。
- 预处理无数据泄漏。
- 能训练 MLP 并保存训练日志。
- 能输出 baseline 与 MLP 的对比指标。
- 能完成至少两个超参数的控制变量实验。
- 能生成论文所需核心图表。

### 9.2 论文验收

- 正文不少于 4000 字。
- 包含监督学习技术横向分析。
- 包含 MLP 数学原理、算法流程、优缺点和个人见解。
- 包含完整的“数据 + 特征 + 模型 + 评估”应用方案。
- 包含训练曲线、混淆矩阵、超参敏感性折线图。
- 包含环境依赖、代码逻辑说明和第三方库版本。
- 包含 baseline 对比和提升幅度分析。
- 包含过拟合判断、错误分析和失败反思。
- 引用 UCI 数据集和相关资料，避免无来源表述。

## 10. 当前默认假设

- 论文和 `overview.md` 使用中文。
- 姓名和学号暂用占位符，最终提交前手动替换。
- PDF 在 Overleaf 云端编译，本地只负责生成 `.tex` 和实验材料。
- 数据采用 UCI Adult，手动下载到本地后由代码处理。
- 主模型为 MLP，深入技术评价也写 MLP。
- Logistic Regression 作为强 baseline，多数类和随机预测作为基础 baseline。
- 当前本机环境为 Windows、Python 3.11.6、PyTorch CPU 版，无 CUDA。
- 不强制安装新依赖，优先使用已存在的 `numpy`、`pandas`、`matplotlib`、`scikit-learn`、`torch`、`tqdm`。
