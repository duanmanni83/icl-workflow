# ICL Workflow: MCP-based Strong Lens ICL Subtraction

一个基于 Model Context Protocol (MCP) 的强透镜星系团内光（ICL）剥离工作流，将传统依赖经验的"开盲盒"过程解耦为原子化的工具调用、状态明确的中间产物以及标准化的决策卡点。

## 核心特性

- **四步标准化工作流**：从初始 Mask 获取到最终 ICL 剥离的完整流程
- **人类在环 (Human-in-the-Loop)**：关键节点自动生成可视化报告，等待人工审批
- **置信度路由 (Confidence Routing)**：根据场复杂度自动选择全自动/半自动/手动模式
- **MCP 兼容**：可作为 MCP Server 部署，支持标准化工具调用

## 架构设计

```
┌─────────────────────────────────────────────────────────────────┐
│                    ICL Workflow Architecture                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Step 0: Field Complexity Assessment                             │
│  ├── 工具: tool_evaluate_field_complexity                        │
│  └── 产出: complexity_score, recommended_mode                    │
│                                                                  │
│  Step 1: Initial Galaxy Mask                                     │
│  ├── 工具: tool_extract_initial_mask (SEP)                       │
│  ├── 产出: seg_map_initial.fits                                  │
│  └── HITL: 检查引力弧是否被误杀                                   │
│                                                                  │
│  Step 2: Diffraction Spike Modeling                              │
│  ├── 工具: tool_generate_spike_mask                              │
│  ├── 产出: spike_mask.fits                                       │
│  └── HITL: 几何对齐校准                                           │
│                                                                  │
│  Step 3: Mask Merge & Dilate                                     │
│  ├── 工具: tool_merge_and_dilate_mask                            │
│  ├── 产出: final_master_mask.fits                                │
│  └── HITL: 信噪比与覆盖度平衡                                     │
│                                                                  │
│  Step 4: ICL Interpolation & Subtraction                         │
│  ├── 工具: tool_interpolate_and_subtract                         │
│  ├── 产出: icl_model.fits, clean_science_residual.fits           │
│  └── HITL: 物理保真度检验（通量守恒 + 形态保持）                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 安装

```bash
# 克隆仓库
cd /Users/duanmanni/Code/ICL_workflow

# 安装依赖
pip install -e ".[dev]"
```

## 快速开始

### 1. 使用 Python API

```python
from src.workflow import ICLWorkflow, WorkflowConfig

# 配置工作流
config = WorkflowConfig(
    output_dir="./outputs",
    detect_thresh=1.5,
    dilation_factor=1.5,
    interpolation_method="rbf"
)

# 创建并运行工作流
workflow = ICLWorkflow(config)

# 注册 HITL 回调
def human_callback(step_name, result):
    print(f"Review {step_name}: {result.visualization_path}")
    return input("Approve? (y/n): ").lower() == 'y'

workflow.register_human_callback(human_callback)

# 运行完整工作流
state = workflow.run(
    "path/to/image.fits",
    instrument="CSST",  # 或 "Euclid", "HST", "JWST"
    mode="auto"         # "auto", "semi_auto", "manual"
)

print(workflow.get_summary())
```

### 2. 使用命令行

```bash
# 查看可用工具
python cli.py tools

# 运行完整工作流
python cli.py workflow image.fits \
    --instrument CSST \
    --mode auto \
    --output ./outputs

# 启动 MCP Server
python cli.py server --stdio
```

### 3. 分步执行

```python
from src.tools import *

# Step 1: 初始 Mask
result1 = ToolExtractInitialMask.execute(
    "image.fits",
    detect_thresh=1.5,
    output_dir="./outputs"
)

# 如有需要，排除特定区域（保护引力弧）
if result1.metrics.get('arc_warning'):
    result_fix = ToolExcludeRegionFromMask.execute(
        result1.output_path,
        regions=[{"x": 100, "y": 100, "radius": 20, "shape": "circle"}],
        output_dir="./outputs"
    )

# Step 2: 星芒 Mask
result2 = ToolGenerateSpikeMask.execute(
    "image.fits",
    instrument="CSST",
    center_x=result1.metrics['center_x'],
    center_y=result1.metrics['center_y'],
    output_dir="./outputs"
)

# Step 3: 合并与膨胀
result3 = ToolMergeAndDilateMask.execute(
    result1.output_path,
    result2.output_path,
    dilation_factor=1.5,
    output_dir="./outputs"
)

# Step 4: ICL 剥离
result4 = ToolInterpolateAndSubtract.execute(
    "image.fits",
    result3.output_path,
    method="rbf",
    output_dir="./outputs"
)

print(f"质量评分: {result4.metrics['overall_score']}/100")
```

## 工具签名参考

| 工具名 | 描述 | 关键参数 |
|--------|------|----------|
| `tool_evaluate_field_complexity` | 场复杂度评估 | `image_fits_path`, `detect_thresh` |
| `tool_extract_initial_mask` | 初始星系 Mask | `detect_thresh`, `min_area` |
| `tool_exclude_region_from_mask` | 排除特定区域 | `mask_path`, `regions` |
| `tool_generate_spike_mask` | 衍射尖峰 Mask | `instrument`, `center_x`, `center_y` |
| `tool_merge_and_dilate_mask` | 合并与膨胀 | `dilation_factor`, `kernel_shape` |
| `tool_interpolate_and_subtract` | ICL 剥离 | `method` (rbf/clough-tocher/linear) |

## 质量评估指标

### 通量守恒 (Flux Conservation)
- **Negative Pixel Ratio**: 残差图中负像素占比
- **Mean/Std Residual**: 残差均值与标准差
- **Flux Score**: 0-100 分，越高越好

### 形态保持 (Shape Preservation)
- **Ellipticity RMSE**: 剥离前后背景源椭率均方根误差
- **Ellipticity Bias**: 椭率系统偏差 (e1, e2)
- **Shape Score**: 0-100 分，越高越好

### 综合评分
- **Overall Score**: 综合得分
- **Passed**: 是否通过质量检验（得分 >= 70, 负像素比 < 0.3, RMSE < 0.1）

## 仪器支持

| 仪器 | 星芒数量 | 默认宽度 | 默认旋转角 |
|------|----------|----------|------------|
| CSST | 4 | 3.0 | 0° |
| Euclid | 6 | 2.5 | 0° |
| HST | 4 | 2.0 | 45° |
| JWST | 6 | 2.0 | 0° |

## 项目结构

```
ICL_workflow/
├── src/
│   ├── __init__.py
│   ├── types.py          # 类型定义
│   ├── utils.py          # FITS 处理、可视化、指标
│   ├── tools.py          # MCP 工具实现
│   ├── workflow.py       # 工作流编排
│   └── mcp_server.py     # MCP Server
├── tests/
│   └── test_tools.py     # 单元测试
├── examples/
│   └── example_usage.py  # 使用示例
├── cli.py                # 命令行入口
├── pyproject.toml        # 项目配置
└── README.md
```

## 开发与测试

```bash
# 运行测试
pytest tests/

# 代码格式化
black src/
ruff check src/

# 类型检查
mypy src/
```

## 许可证

MIT License

## 引用

如果本工作流对您的研究有帮助，请引用：

```
ICL Workflow: An MCP-based Framework for Strong Lens ICL Subtraction
Duan Manni, 2025
```
