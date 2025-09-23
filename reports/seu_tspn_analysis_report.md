# SEU TSPN 解释性分析总结（待生成）

运行 `post/seu_tspn_analysis.py` 并提供 SEU 数据集与训练好的模型权重后，本报告会被自动覆盖，包含以下内容：

- 故障类别、转速与载荷条件下的模态贡献热图解读。
- 依据能量型、冲击型、复杂度型分组的 13 项指标雷达图说明。
- 模态贡献与各类指标之间的偏相关矩阵分析。
- 各故障类别特异性的模态—指标关联结论。

示例命令：

```bash
python post/seu_tspn_analysis.py \
    --data /path/to/SEU_bearing_20Hz_2_data.npy \
    --labels /path/to/SEU_bearing_20Hz_2_label.npy \
    --speed-labels /path/to/SEU_bearing_20Hz_2_speed.npy \
    --load-labels /path/to/SEU_bearing_20Hz_2_load.npy \
    --checkpoint save/test/model_seu/tspn.ckpt \
    --config configs/a_010_SEU/config_basic.yaml
```

命令执行完成后，可在 `save/figure/seu/analysis/` 查看生成的热图与雷达图。
