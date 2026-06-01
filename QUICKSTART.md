# ICL Workflow 快速开始指南

## 1. 环境准备

```bash
cd /Users/duanmanni/Code/ICL_workflow
pip install -e ".[dev]"
```

## 2. 生成测试数据

```bash
# 生成包含引力弧和衍射星芒的测试图像
python examples/generate_test_data.py --output examples/test_image.fits

# 查看生成的图像
python -c "from astropy.io import fits; import matplotlib.pyplot as plt; d=fits.getdata('examples/test_image.fits'); plt.imshow(d, norm=plt.matplotlib.colors.LogNorm()); plt.colorbar(); plt.savefig('test_preview.png')"
```

## 3. 运行演示

### 方式一：分步执行（推荐用于理解流程）

```bash
python examples/demo_workflow.py --step
```

这会展示每个步骤的输出和中间结果。

### 方式二：完整工作流 API

```bash
python examples/demo_workflow.py --full
```

这会展示如何使用高级 API 一次性运行完整流程。

## 4. 使用命令行工具

```bash
# 查看帮助
python cli.py --help

# 列出可用工具
python cli.py tools

# 运行完整工作流（如果有真实数据）
python cli.py workflow path/to/your/image.fits \
    --instrument CSST \
    --mode auto \
    --output ./my_outputs
```

## 5. Python API 快速入门

```python
from src.workflow import ICLWorkflow, WorkflowConfig

# 配置
config = WorkflowConfig(
    output_dir="./my_outputs",
    detect_thresh=1.5,
    dilation_factor=1.5,
    interpolation_method="rbf"
)

# 创建工作流
workflow = ICLWorkflow(config)

# 注册人工审核回调（可选）
def review_callback(step_name, result):
    print(f"Review: {step_name}")
    print(f"Viz: {result.visualization_path}")
    return input("Approve? (y/n): ").lower() == 'y'

workflow.register_human_callback(review_callback)

# 运行工作流
state = workflow.run(
    "path/to/image.fits",
    instrument="CSST",
    mode="auto"  # auto, semi_auto, manual
)

# 打印结果
print(workflow.get_summary())
```

## 6. MCP Server 集成

```bash
# 启动 MCP Server（stdio 模式，供 MCP 客户端使用）
python cli.py server --stdio

# 或使用 Python API
from src.mcp_server import ICLWorkflowServer

server = ICLWorkflowServer()

# 查看能力
print(server.get_capabilities())

# 执行工具
result = server.execute_tool("tool_extract_initial_mask", {
    "image_fits_path": "image.fits",
    "detect_thresh": 1.5
})
print(result)
```

## 7. 工作流程详解

```
┌────────────────────────────────────────────────────────────────┐
│                    ICL 剥离工作流                               │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Step 0: 场复杂度评估                                            │
│  ├── 输入: FITS 图像                                             │
│  └── 产出: 复杂度评分, 推荐模式 (auto/semi_auto/manual)           │
│                                                                 │
│  Step 1: 初始星系 Mask (SEP)                                    │
│  ├── 输入: FITS 图像                                             │
│  ├── 参数: detect_thresh=1.5, min_area=5                         │
│  ├── 产出: seg_map_initial.fits                                  │
│  └── HITL: 检查引力弧是否被误杀                                   │
│      └── 可视化: step1_mask_overlay.png                         │
│                                                                 │
│  Step 2: 衍射尖峰 Mask                                           │
│  ├── 输入: FITS 图像, 仪器类型, 中心坐标                           │
│  ├── 仪器: CSST(4星芒), Euclid(6星芒), HST(4星芒, 45°), JWST      │
│  ├── 产出: spike_mask.fits                                       │
│  └── HITL: 验证星芒角度对齐                                       │
│      └── 可视化: step2_spike_mask.png                           │
│                                                                 │
│  Step 3: Mask 合并与膨胀                                         │
│  ├── 输入: seg_map + spike_mask                                  │
│  ├── 参数: dilation_factor=1.5, kernel_shape="disk"              │
│  ├── 产出: final_master_mask.fits                                │
│  └── HITL: 检查膨胀是否吞没背景源                                 │
│      └── 可视化: step3_dilation.png                             │
│                                                                 │
│  Step 4: ICL 插值与剥离                                          │
│  ├── 输入: 原图 + final_mask                                     │
│  ├── 方法: rbf(推荐), clough-tocher, linear, nearest             │
│  ├── 产出: icl_model.fits, clean_science_residual.fits           │
│  └── HITL: 物理保真度检验                                         │
│      ├── 通量守恒: 负像素比, 均值残差                              │
│      ├── 形态保持: 背景源椭率 RMSE                                │
│      └── 可视化: step4_residual_analysis.png                    │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

## 8. 输出文件说明

| 文件 | 说明 |
|------|------|
| `seg_map_initial.fits` | 初始 SEP 分割图 |
| `spike_mask.fits` | 衍射尖峰 Mask |
| `final_master_mask.fits` | 合并后的最终 Mask |
| `icl_model.fits` | 估算的 ICL 本底模型 |
| `clean_science_residual.fits` | 剥离后的干净科学图像 |
| `step*_*.png` | 各步骤的可视化报告 |

## 9. 质量评估指标

- **Overall Score**: 综合评分 (0-100)
- **Flux Conservation**: 通量守恒评分
  - Negative Pixel Ratio: 残差中负像素占比（越低越好）
- **Shape Preservation**: 形态保持评分
  - Ellipticity RMSE: 剥离前后背景源椭率均方根误差（越低越好）
- **Passed**: 是否通过质量检验

## 10. 故障排除

### SEP 安装问题
```bash
# macOS
brew install libomp
pip install sep --no-cache-dir

# Linux
pip install sep
```

### 内存不足
```python
# 对大图像使用分块处理
# 修改 tool_interpolate_and_subtract 中的 chunk_size
```

### 质量检验失败
```python
# 调整参数重新运行
config = WorkflowConfig(
    detect_thresh=2.0,      # 提高检测阈值
    dilation_factor=1.2,    # 降低膨胀系数
    interpolation_method="clough-tocher"  # 换插值方法
)
```

## 11. 下一步

- 查看完整文档: [README.md](README.md)
- 运行测试: `pytest tests/`
- 自定义工具: 修改 `src/tools.py`
- 集成到您的数据处理流程
