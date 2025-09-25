# SEU TSPN 解释性分析总结

## 模态贡献概览
- Class 0 最依赖 J3 axis (平均贡献 0.27).
- Class 1 最依赖 J3 axis (平均贡献 0.28).
- Class 2 最依赖 J3 axis (平均贡献 0.28).
- Class 3 最依赖 J3 axis (平均贡献 0.28).

## 雷达图指标分布
各故障类别的 13 项指标已按能量型、冲击型与复杂度型进行着色区分。归一化雷达图突出显示了每类样本在指标空间的差异。

## 模态-指标偏相关
- J3 axis 模态贡献与 ClearanceFactor (负相关 -0.56)、Max (正相关 0.46) 关联最强。
- U phase 模态贡献与 ClearanceFactor (正相关 0.42)、Max (负相关 -0.24) 关联最强。
- V phase 模态贡献与 Std (正相关 0.33)、ClearanceFactor (负相关 -0.33) 关联最强。
- W phase 模态贡献与 ClearanceFactor (正相关 0.48)、Max (负相关 -0.42) 关联最强。
- D axis 模态贡献与 ClearanceFactor (正相关 0.46)、Max (负相关 -0.35) 关联最强。

## 类特异性模态-指标通路
### Stuck 1
- J3 axis 模态贡献与 ClearanceFactor (负相关 -0.60)、Max (正相关 0.56) 关联最强。
- U phase 模态贡献与 ClearanceFactor (正相关 0.63)、Skewness (正相关 0.54) 关联最强。
- V phase 模态贡献与 ClearanceFactor (负相关 -0.58)、Std (正相关 0.51) 关联最强。
- W phase 模态贡献与 ClearanceFactor (正相关 0.68)、Max (负相关 -0.56) 关联最强。
- D axis 模态贡献与 ClearanceFactor (正相关 0.57)、Max (负相关 -0.40) 关联最强。

### Stuck 2
- J3 axis 模态贡献与 ClearanceFactor (负相关 -0.76)、Max (正相关 0.67) 关联最强。
- U phase 模态贡献与 ClearanceFactor (正相关 0.67)、Kurtosis (正相关 0.52) 关联最强。
- V phase 模态贡献与 Max (正相关 0.57)、ClearanceFactor (负相关 -0.57) 关联最强。
- W phase 模态贡献与 ClearanceFactor (正相关 0.68)、Max (负相关 -0.66) 关联最强。
- D axis 模态贡献与 ClearanceFactor (正相关 0.67)、Max (负相关 -0.66) 关联最强。

### Norm
- J3 axis 模态贡献与 CrestFactor (负相关 -0.60)、RMS (负相关 -0.48) 关联最强。
- U phase 模态贡献与 CrestFactor (正相关 0.52)、RMS (正相关 0.39) 关联最强。
- V phase 模态贡献与 Kurtosis (负相关 -0.34)、Max (负相关 -0.31) 关联最强。
- W phase 模态贡献与 CrestFactor (正相关 0.41)、RMS (正相关 0.21) 关联最强。
- D axis 模态贡献与 CrestFactor (正相关 0.55)、AbsMean (正相关 0.29) 关联最强。

### Ball
- J3 axis 模态贡献与 Var (负相关 -0.37)、CrestFactor (负相关 -0.34) 关联最强。
- U phase 模态贡献与 Skewness (负相关 -0.51)、Max (正相关 0.50) 关联最强。
- V phase 模态贡献与 Skewness (正相关 0.67)、Max (负相关 -0.57) 关联最强。
- W phase 模态贡献与 CrestFactor (正相关 0.59)、Min (正相关 0.54) 关联最强。
- D axis 模态贡献与 Skewness (负相关 -0.55)、Mean (负相关 -0.34) 关联最强。
