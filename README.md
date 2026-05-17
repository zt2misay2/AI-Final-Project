# UCI Adult 收入二分类期末项目

本项目按 `overview.md` 的路线完成 UCI Adult 数据集二分类实验，包含数据清洗、EDA、baseline、MLP、超参数敏感性分析和 Overleaf 论文源码。

项目仓库：<https://github.com/zt2misay2/AI-Final-Project>

## 运行顺序

```powershell
python src\prepare_data.py
python src\train_baselines.py
python src\train_mlp.py --epochs 50 --lr 0.001 --dropout 0.3 --batch-size 64
python src\analyze_hparams.py
python src\plot_results.py
```

原始数据应放在 `data/raw/`：

- `adult.data`
- `adult.test`
- `adult.names`

本仓库当前已从本地 `adult/` 目录复制到 `data/raw/`。官方来源为 <https://archive.ics.uci.edu/ml/machine-learning-databases/adult/>。

## 主要输出

- `results/baseline_results.csv`
- `results/mlp_training_log.csv`
- `results/mlp_test_metrics.csv`
- `results/hparam_results.csv`
- `figures/*.png`
- `models/mlp_best.pt`
- `paper/main.tex`
- `paper/references.bib`

论文使用 `ctexart`，建议在 Overleaf 选择 XeLaTeX 编译。姓名、学号仍是占位符，提交前需要替换。
