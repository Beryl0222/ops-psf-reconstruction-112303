# 曝光换档下的空间变 PSF 重建算子

`data/record.json` 是 曝光换档下的空间变 PSF 重建算子 使用的本地合成交换样本。数组按 JSON 的下标顺序排列。本文的坐标、单位和枚举约定与该记录配套。记录中如果有 `scale` 或 `limits`，对应同一输入协议的部署规格。

网格坐标是工件上的像素中心，两个轴按行列顺序。三个 kernel_basis 是以中间元素为偏移零点的单位和响应。calibrations 的末轴是给这三个核的线性系数，观测中心坐标上用相邻网格节点的双线性权重。偏移表示用于预测该观测的输入坐标相对观测中心的位置。origin 是条带左上采样坐标，invalid 是条带内部相对坐标，noise_variance 对应预测余量方差。替换只接受 ID 相同且版本更高的帧，版本相同但内容不同属于输入冲突。gradient_probe 的张量是对原始预测值的导数，权重由调用方在上游余量中表达。

## 读取样本

使用 Python 3.11 或 3.12，并在虚拟环境安装锁定依赖。读取入口返回原始记录。作为命令运行时，它会列出各顶层输入项目以及 NumPy 版本。

```sh
python -m pip install -r requirements.txt
python -m psf_reconstruction_input
python -m unittest discover -s tests -v
```

测试读取交换样本，核对其中的张量尺寸、标识与数据约定。
