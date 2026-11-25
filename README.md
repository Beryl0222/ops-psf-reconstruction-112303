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

## 可调用算子与重建

主要稳定入口是 `psf_reconstruction`（另提供 `spatial_psf` 与 `solution` 别名模块）：

```python
import psf_reconstruction as psf

record = psf.load_record()

# 当前帧集合：每个 id 只保留最高 revision；也可用 which="initial" 取锁定旧帧。
op = psf.build_operator(record, which="current")
locked_op = psf.build_operator(record, which="initial")

result = op.reconstruct(
    smoothness=record["smoothness"],       # λ：一阶差分二次项系数
    tolerance=record["relative_tolerance"], # 法方程残差的相对容限
    max_iterations=record["iterations"],
)

result.converged        # False 时 result.image 仍是可查看的未收敛候选
result.directly_observed
roi = result.inclusive_window(record["requested_roi"])
roi.missing_direct_observation  # 平滑出数值但没有直接传感器响应的位置
roi.missing_coordinates

# 原始 A/A^T 严格伴随；加权拟合使用 A^T M W r。
pred = op.forward(result.image)
back = op.adjoint([...])
weighted_back = op.weighted_adjoint([...])
assert op.dot_test()["weighted_abs_error"] < 1e-10
```

帧替换、锁定比较和迟到梯度请走 `ScanSession`：

```python
workflow = psf.run_record_workflow(record)
workflow.locked_result      # over-a@1(slow) + over-b@1(fast)
workflow.current_result     # over-a@1(slow) + over-b@2(slow)
workflow.comparison["difference"]
workflow.gradient.gradient  # 固定在锁定图和 over-b@1/fast 原调用上
```

`FrameRegistry` 只接受相同 ID 的更高 revision；同 revision 但像素、掩码、
档位、原点或噪声不同会抛 `FrameConflictError`。旧 revision 仍保存在历史中，
因此 `["over-b", 1]` 的迟到梯度始终使用原调用的 `fast` 标定，不会自动改用
当前 `over-b@2` 的 `slow` 标定。

### 内存约定

实现不构造稠密观测矩阵，也不一次展开所有条带的空间变核。每次矩阵向量乘积
逐帧处理，并按 `scale.tile_rows` / `scale.scratch_bytes` 对帧分行切片；
常驻的大数组主要是工件网格向量。`SpatialPSFOperator.as_linear_operator()`
还提供 SciPy 的无矩阵 `LinearOperator` 视图。

### ROI 约定

`requested_roi = [y0, x0, y1, x1]`。`inclusive_window()` 把末端视为包含
边界（样本中的 `[1,2,6,8]` 即到最后一行/一列），`half_open_window()` 按
Python 半开区间处理。两种窗口都会返回直接观测/缺失直接观测掩码。
