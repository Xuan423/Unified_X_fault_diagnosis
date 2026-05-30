# SEU TSPN 解释性分析总结

## 模态贡献概览
- Class 0 最依赖 Torque (平均贡献 0.52).
- Class 1 最依赖 Torque (平均贡献 0.51).
- Class 2 最依赖 Torque (平均贡献 0.52).
- Class 3 最依赖 Torque (平均贡献 0.52).
- Class 4 最依赖 Torque (平均贡献 0.52).

## 雷达图指标分布
各故障类别的 13 项指标已按能量型、冲击型与复杂度型进行着色区分。归一化雷达图突出显示了每类样本在指标空间的差异。

## 模态-指标偏相关
- Torque 模态贡献与 AbsMean (正相关 0.29)、RMS (正相关 0.21) 关联最强。
- Vibration 模态贡献与 AbsMean (负相关 -0.29)、RMS (负相关 -0.21) 关联最强。

## 类特异性模态-指标通路
### Class 0
- Torque 模态贡献与 AbsMean (正相关 0.58)、RMS (正相关 0.45) 关联最强。
- Vibration 模态贡献与 AbsMean (负相关 -0.58)、RMS (负相关 -0.45) 关联最强。

### Class 1
- Torque 模态贡献与 CrestFactor (负相关 -0.76)、ShapeFactor (负相关 -0.70) 关联最强。
- Vibration 模态贡献与 CrestFactor (正相关 0.76)、ShapeFactor (正相关 0.70) 关联最强。

### Class 2
- Torque 模态贡献与 RMS (负相关 -0.62)、Kurtosis (负相关 -0.46) 关联最强。
- Vibration 模态贡献与 RMS (正相关 0.62)、Kurtosis (正相关 0.46) 关联最强。

### Class 3
- Torque 模态贡献与 AbsMean (正相关 0.34)、Entropy (负相关 -0.22) 关联最强。
- Vibration 模态贡献与 AbsMean (负相关 -0.34)、Entropy (正相关 0.22) 关联最强。

### Class 4
- Torque 模态贡献与 Skewness (正相关 0.58)、RMS (正相关 0.52) 关联最强。
- Vibration 模态贡献与 Skewness (负相关 -0.58)、RMS (负相关 -0.52) 关联最强。
