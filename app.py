#!/usr/bin/env python3
"""
ICL Workflow - Human-in-the-Loop Visual System
基于Streamlit的可视化人机交互系统
"""

import streamlit as st
import sys
from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.visualization import simple_norm, LogStretch, LinearStretch
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import io

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent / "src"))
from tools import (
    ToolExtractInitialMask,
    ToolGenerateSpikeMask,
    ToolMergeAndDilateMask,
    ToolInterpolateAndSubtract,
    ToolEvaluateFieldComplexity,
    ToolExcludeRegionFromMask,
)
from core_types import WorkflowState

# 页面配置
st.set_page_config(
    page_title="ICL Workflow - HITL",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .step-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #ff7f0e;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .approval-btn {
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """初始化会话状态"""
    if 'workflow_state' not in st.session_state:
        st.session_state.workflow_state = {
            'current_step': 0,  # 0-5
            'image_path': None,
            'instrument': 'HST',
            'step_results': {},  # 存储每步结果
            'approved_steps': set(),

            # Step 1 参数
            'detect_thresh': 1.5,
            'min_area': 5,
            'excluded_regions': [],  # 手动排除的区域

            # Step 2 参数
            'center_x': None,
            'center_y': None,
            'spike_width': 3.0,
            'spike_length': 150.0,
            'rotation_angle': 0.0,

            # Step 3 参数
            'dilation_factor': 1.5,
            'kernel_shape': 'disk',

            # Step 4 参数
            'interpolation_method': 'rbf',
        }


def display_fits_image(data, title, colormap='gray', log_scale=True):
    """显示FITS图像"""
    fig, ax = plt.subplots(figsize=(8, 8))

    if log_scale and np.all(data > 0):
        norm = simple_norm(data, 'log', percent=99)
    else:
        norm = simple_norm(data, 'linear', percent=99)

    im = ax.imshow(data, norm=norm, cmap=colormap, origin='lower')
    ax.set_title(title, fontsize=12)
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.axis('off')

    return fig


def display_comparison(original, processed, title1="Original", title2="Processed"):
    """并排显示对比图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for ax, data, title in [(ax1, original, title1), (ax2, processed, title2)]:
        if np.all(data > 0):
            norm = simple_norm(data, 'log', percent=99)
        else:
            norm = simple_norm(data, 'linear', percent=99)
        im = ax.imshow(data, norm=norm, cmap='gray', origin='lower')
        ax.set_title(title, fontsize=12)
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.axis('off')

    plt.tight_layout()
    return fig


def step_0_file_upload():
    """Step 0: 文件上传和场复杂度评估"""
    st.markdown('<div class="step-header">Step 0: 数据上传与评估</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("上传FITS文件", type=['fits', 'fit'])

    if uploaded_file is not None:
        # 保存上传的文件
        save_path = Path("uploads") / uploaded_file.name
        save_path.parent.mkdir(exist_ok=True)
        save_path.write_bytes(uploaded_file.getvalue())

        st.session_state.workflow_state['image_path'] = str(save_path)

        # 读取并显示预览
        with fits.open(save_path) as hdul:
            data = hdul[0].data
            if data is None and len(hdul) > 1:
                data = hdul[1].data

            st.session_state.workflow_state['image_shape'] = data.shape

            col1, col2 = st.columns([2, 1])

            with col1:
                fig = display_fits_image(data, "Input Image Preview")
                st.pyplot(fig)

            with col2:
                st.write("**图像信息**")
                st.write(f"尺寸: {data.shape}")
                st.write(f"数据类型: {data.dtype}")
                st.write(f"数值范围: [{data.min():.2e}, {data.max():.2e}]")

                # 选择仪器
                instrument = st.selectbox(
                    "选择望远镜/仪器",
                    ['HST', 'CSST', 'Euclid', 'JWST'],
                    key='instrument_select'
                )
                st.session_state.workflow_state['instrument'] = instrument

        # 场复杂度评估
        if st.button("评估场复杂度", key='eval_complexity'):
            with st.spinner("分析中..."):
                result = ToolEvaluateFieldComplexity.execute(
                    str(save_path),
                    output_dir="outputs/app"
                )

                st.session_state.workflow_state['step_results'][0] = result

                # 检查是否成功
                if not result.success or result.metrics is None:
                    st.error(f"场复杂度评估失败: {result.message}")
                    st.info("提示: JWST数据可能需要预处理。尝试将图像转换为float64格式，或检查数据是否包含NaN/Inf值。")
                else:
                    # 安全获取metrics值
                    metrics = result.metrics
                    complexity_score = metrics.get('complexity_score', 0)
                    num_bright = metrics.get('num_bright_stars', 0)
                    num_faint = metrics.get('num_faint_stars', 0)
                    recommended_auto = metrics.get('recommended_auto_mode', False)

                    # 显示结果
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    cols = st.columns(4)
                    with cols[0]:
                        st.metric("复杂度评分", f"{complexity_score:.1f}")
                    with cols[1]:
                        st.metric("亮星数量", int(num_bright))
                    with cols[2]:
                        st.metric("暗星数量", int(num_faint))
                    with cols[3]:
                        mode = "AUTO" if recommended_auto else "MANUAL"
                        st.metric("推荐模式", mode)
                    st.markdown('</div>', unsafe_allow_html=True)

                    st.info(result.message)

                # 下一步按钮
                if st.button("进入 Step 1 →", key='to_step1'):
                    st.session_state.workflow_state['current_step'] = 1
                    st.rerun()


def step_1_initial_mask():
    """Step 1: 初始Mask提取（可调参数）"""
    st.markdown('<div class="step-header">Step 1: 初始星系Mask提取</div>', unsafe_allow_html=True)

    image_path = st.session_state.workflow_state['image_path']

    # 参数控制面板
    with st.sidebar:
        st.subheader("⚙️ Step 1 参数")

        detect_thresh = st.slider(
            "检测阈值 (sigma)",
            min_value=0.5,
            max_value=5.0,
            value=st.session_state.workflow_state['detect_thresh'],
            step=0.1,
            key='step1_thresh'
        )

        min_area = st.slider(
            "最小像素面积",
            min_value=1,
            max_value=50,
            value=st.session_state.workflow_state['min_area'],
            step=1,
            key='step1_minarea'
        )

        st.session_state.workflow_state['detect_thresh'] = detect_thresh
        st.session_state.workflow_state['min_area'] = min_area

        # 运行按钮
        run_button = st.button("🚀 运行提取", key='run_step1')

    # 主显示区域
    if run_button or 1 in st.session_state.workflow_state['step_results']:
        with st.spinner("运行SEP源提取..."):
            result = ToolExtractInitialMask.execute(
                image_path,
                detect_thresh=detect_thresh,
                min_area=min_area,
                output_dir="outputs/app",
                check_region_size=100
            )

            st.session_state.workflow_state['step_results'][1] = result

            if result.success:
                # 读取结果
                mask = fits.getdata(result.output_path)
                original = fits.getdata(image_path)

                # 显示指标
                cols = st.columns(4)
                with cols[0]:
                    st.metric("检测天体数", result.metrics['num_objects'])
                with cols[1]:
                    st.metric("中心X", f"{result.metrics['center_x']:.1f}")
                with cols[2]:
                    st.metric("中心Y", f"{result.metrics['center_y']:.1f}")
                with cols[3]:
                    if result.metrics['arc_warning']:
                        st.error("⚠️ 可能存在弧污染")
                    else:
                        st.success("✓ 无明显弧污染")

                # 保存中心坐标
                st.session_state.workflow_state['center_x'] = result.metrics['center_x']
                st.session_state.workflow_state['center_y'] = result.metrics['center_y']

                # 显示图像
                tab1, tab2, tab3 = st.tabs(["原图+Mask", "仅Mask", "可视化报告"])

                with tab1:
                    fig, ax = plt.subplots(figsize=(10, 10))
                    norm = simple_norm(original, 'log', percent=99)
                    ax.imshow(original, norm=norm, cmap='gray', origin='lower')
                    ax.contour(mask, levels=[0.5], colors='red', linewidths=1.5, alpha=0.7)
                    ax.set_title("Original with Mask Overlay")
                    ax.axis('off')
                    st.pyplot(fig)

                with tab2:
                    fig = display_fits_image(mask, "Segmentation Mask", colormap='Reds', log_scale=False)
                    st.pyplot(fig)

                with tab3:
                    if result.visualization_path and Path(result.visualization_path).exists():
                        st.image(result.visualization_path)

                # 人工审核区域
                st.markdown("---")
                st.subheader("人工审核")

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✅ 批准 - 进入下一步", key='approve_step1'):
                        st.session_state.workflow_state['approved_steps'].add(1)
                        st.session_state.workflow_state['current_step'] = 2
                        st.rerun()

                with col2:
                    if st.button("❌ 拒绝 - 调整参数", key='reject_step1'):
                        st.warning("请在左侧调整参数后重新运行")

                # 手动排除区域功能
                with st.expander("🔧 手动排除区域（保护引力弧）"):
                    st.write("如果引力弧被错误mask，可在此排除特定区域")
                    exclude_x = st.number_input("中心X", value=0, step=1)
                    exclude_y = st.number_input("中心Y", value=0, step=1)
                    exclude_r = st.number_input("半径", value=20, step=1)

                    if st.button("添加排除区域"):
                        st.session_state.workflow_state['excluded_regions'].append({
                            'x': exclude_x, 'y': exclude_y, 'radius': exclude_r, 'shape': 'circle'
                        })
                        st.success(f"已添加排除区域: ({exclude_x}, {exclude_y}), r={exclude_r}")


def step_2_spike_mask():
    """Step 2: 衍射星芒Mask"""
    st.markdown('<div class="step-header">Step 2: 衍射星芒Mask生成</div>', unsafe_allow_html=True)

    image_path = st.session_state.workflow_state['image_path']
    instrument = st.session_state.workflow_state['instrument']
    cx = st.session_state.workflow_state['center_x']
    cy = st.session_state.workflow_state['center_y']

    # 参数控制
    with st.sidebar:
        st.subheader("⚙️ Step 2 参数")

        spike_width = st.slider("星芒宽度", 1.0, 10.0,
                               st.session_state.workflow_state.get('spike_width', 3.0), 0.5)
        spike_length = st.slider("星芒长度", 50.0, 300.0,
                                st.session_state.workflow_state.get('spike_length', 150.0), 10.0)
        rotation_angle = st.slider("旋转角度", -45.0, 45.0,
                                  st.session_state.workflow_state.get('rotation_angle', 0.0), 1.0)

        st.session_state.workflow_state.update({
            'spike_width': spike_width,
            'spike_length': spike_length,
            'rotation_angle': rotation_angle
        })

        run_button = st.button("🚀 生成星芒Mask", key='run_step2')

    # 显示中心位置
    st.write(f"**透镜星系中心**: ({cx:.1f}, {cy:.1f})")
    st.write(f"**仪器**: {instrument}")

    if run_button or 2 in st.session_state.workflow_state['step_results']:
        with st.spinner("生成星芒Mask..."):
            result = ToolGenerateSpikeMask.execute(
                image_path,
                instrument=instrument,
                center_x=cx,
                center_y=cy,
                spike_width=spike_width,
                spike_length=spike_length,
                rotation_angle=rotation_angle,
                output_dir="outputs/app"
            )

            st.session_state.workflow_state['step_results'][2] = result

            if result.success:
                # 读取结果
                spike_mask = fits.getdata(result.output_path)
                original = fits.getdata(image_path)

                # 显示参数
                cols = st.columns(4)
                with cols[0]:
                    st.metric("星芒数量", result.metrics['num_spikes'])
                with cols[1]:
                    st.metric("宽度", f"{result.metrics['spike_width']:.1f}pix")
                with cols[2]:
                    st.metric("长度", f"{result.metrics['spike_length']:.1f}pix")
                with cols[3]:
                    st.metric("旋转角", f"{result.metrics['rotation_angle']:.1f}°")

                # 显示图像
                tab1, tab2 = st.tabs(["星芒叠加", "可视化报告"])

                with tab1:
                    fig, ax = plt.subplots(figsize=(10, 10))
                    norm = simple_norm(original, 'log', percent=99)
                    ax.imshow(original, norm=norm, cmap='gray', origin='lower')
                    ax.contour(spike_mask, levels=[0.5], colors='lime', linewidths=2, alpha=0.7)
                    ax.plot(cx, cy, 'r+', markersize=20, markeredgewidth=2)
                    ax.set_title("Spike Mask Overlay")
                    ax.axis('off')
                    st.pyplot(fig)

                with tab2:
                    if result.visualization_path and Path(result.visualization_path).exists():
                        st.image(result.visualization_path)

                # 审核
                st.markdown("---")
                st.subheader("人工审核")
                st.write("检查星芒Mask是否与实际衍射星芒对齐")

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✅ 批准 - 进入下一步", key='approve_step2'):
                        st.session_state.workflow_state['approved_steps'].add(2)
                        st.session_state.workflow_state['current_step'] = 3
                        st.rerun()

                with col2:
                    if st.button("❌ 拒绝 - 调整参数", key='reject_step2'):
                        st.warning("请在左侧调整参数后重新运行")


def step_3_merge_dilate():
    """Step 3: Mask合并与膨胀"""
    st.markdown('<div class="step-header">Step 3: Mask合并与膨胀</div>', unsafe_allow_html=True)

    image_path = st.session_state.workflow_state['image_path']

    # 获取前两步的结果路径
    step1_result = st.session_state.workflow_state['step_results'].get(1)
    step2_result = st.session_state.workflow_state['step_results'].get(2)

    if not step1_result or not step2_result:
        st.error("请先完成Step 1和Step 2")
        return

    # 参数控制
    with st.sidebar:
        st.subheader("⚙️ Step 3 参数")

        dilation_factor = st.slider("膨胀系数", 0.5, 3.0,
                                   st.session_state.workflow_state['dilation_factor'], 0.1)
        kernel_shape = st.selectbox("核形状", ['disk', 'box'],
                                   index=0 if st.session_state.workflow_state['kernel_shape'] == 'disk' else 1)

        st.session_state.workflow_state.update({
            'dilation_factor': dilation_factor,
            'kernel_shape': kernel_shape
        })

        run_button = st.button("🚀 合并与膨胀", key='run_step3')

    if run_button or 3 in st.session_state.workflow_state['step_results']:
        with st.spinner("合并和膨胀Mask..."):
            result = ToolMergeAndDilateMask.execute(
                step1_result.output_path,
                step2_result.output_path,
                dilation_factor=dilation_factor,
                kernel_shape=kernel_shape,
                image_fits_path=image_path,
                output_dir="outputs/app"
            )

            st.session_state.workflow_state['step_results'][3] = result

            if result.success:
                # 显示指标
                cols = st.columns(3)
                with cols[0]:
                    st.metric("原始覆盖率", f"{result.metrics['original_coverage']:.1f}%")
                with cols[1]:
                    st.metric("膨胀后覆盖率", f"{result.metrics['dilated_coverage']:.1f}%")
                with cols[2]:
                    increase = result.metrics['dilated_coverage'] - result.metrics['original_coverage']
                    st.metric("增加", f"{increase:.1f}%", delta=f"+{increase:.1f}")

                # 显示可视化
                if result.visualization_path and Path(result.visualization_path).exists():
                    st.image(result.visualization_path)

                # 警告
                if result.metrics['dilated_coverage'] > 30:
                    st.warning("⚠️ 覆盖率过高，可能吞没了背景源，建议降低膨胀系数")

                # 审核
                st.markdown("---")
                st.subheader("人工审核")
                st.write("检查膨胀后的Mask是否合适")

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✅ 批准 - 进入下一步", key='approve_step3'):
                        st.session_state.workflow_state['approved_steps'].add(3)
                        st.session_state.workflow_state['current_step'] = 4
                        st.rerun()

                with col2:
                    if st.button("❌ 拒绝 - 调整参数", key='reject_step3'):
                        st.warning("请在左侧调整参数后重新运行")


def step_4_interpolation():
    """Step 4: ICL插值与剥离"""
    st.markdown('<div class="step-header">Step 4: ICL插值与剥离</div>', unsafe_allow_html=True)

    image_path = st.session_state.workflow_state['image_path']
    step3_result = st.session_state.workflow_state['step_results'].get(3)

    if not step3_result:
        st.error("请先完成Step 3")
        return

    # 参数控制
    with st.sidebar:
        st.subheader("⚙️ Step 4 参数")

        method = st.selectbox(
            "插值方法",
            ['rbf', 'clough-tocher', 'linear', 'nearest'],
            index=['rbf', 'clough-tocher', 'linear', 'nearest'].index(
                st.session_state.workflow_state['interpolation_method']
            ),
            help="RBF: 质量最高但慢, Linear: 快速, Nearest: 最快"
        )

        st.session_state.workflow_state['interpolation_method'] = method

        run_button = st.button("🚀 运行ICL剥离", key='run_step4')

    if run_button or 4 in st.session_state.workflow_state['step_results']:
        with st.spinner("运行ICL插值与剥离（可能需要几分钟）..."):
            result = ToolInterpolateAndSubtract.execute(
                image_path,
                step3_result.output_path,
                method=method,
                output_dir="outputs/app"
            )

            st.session_state.workflow_state['step_results'][4] = result

            if result.success:
                metrics = result.metrics

                # 显示质量指标
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                cols = st.columns(4)
                with cols[0]:
                    st.metric("综合评分", f"{metrics['overall_score']:.1f}/100")
                with cols[1]:
                    st.metric("通量守恒", f"{metrics['flux_conservation_score']:.1f}/100")
                with cols[2]:
                    st.metric("形态保持", f"{metrics['shape_preservation_score']:.1f}/100")
                with cols[3]:
                    status = "✅ 通过" if metrics['passed'] else "❌ 未通过"
                    st.metric("状态", status)
                st.markdown('</div>', unsafe_allow_html=True)

                # 详细指标
                with st.expander("📊 详细质量指标"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**通量守恒**")
                        st.write(f"- 负像素比例: {metrics['negative_pixel_ratio']:.4f}")
                        st.write(f"- 残差均值: {metrics['mean_residual']:.4f}")
                        st.write(f"- 残差标准差: {metrics['std_residual']:.4f}")
                    with col2:
                        st.write("**形态保持**")
                        st.write(f"- 椭率RMSE: {metrics['ellipticity_rmse']:.4f}")
                        st.write(f"- e1偏差: {metrics['ellipticity_bias_x']:.4f}")
                        st.write(f"- e2偏差: {metrics['ellipticity_bias_y']:.4f}")

                # 显示可视化报告
                if result.visualization_path and Path(result.visualization_path).exists():
                    st.image(result.visualization_path, use_column_width=True)

                # 最终审核
                st.markdown("---")
                st.subheader("最终审核")

                if metrics['passed']:
                    st.success("✅ 质量检验通过！结果可用于科学分析。")
                else:
                    st.error("❌ 质量检验未通过，建议：")
                    st.write("- 尝试不同的插值方法（推荐RBF）")
                    st.write("- 调整膨胀系数")
                    st.write("- 检查是否有重要结构被mask")

                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("✅ 完成工作流", key='approve_step4'):
                        st.session_state.workflow_state['approved_steps'].add(4)
                        st.session_state.workflow_state['current_step'] = 5
                        st.rerun()

                with col2:
                    if st.button("🔙 返回调整", key='back_to_adjust'):
                        st.session_state.workflow_state['current_step'] = 3
                        st.rerun()

                with col3:
                    # 下载结果按钮
                    if result.output_path and Path(result.output_path).exists():
                        with open(result.output_path, 'rb') as f:
                            st.download_button(
                                "📥 下载残差图",
                                f,
                                file_name="clean_science_residual.fits",
                                mime="application/fits"
                            )


def step_5_summary():
    """Step 5: 工作流总结"""
    st.markdown('<div class="step-header">工作流完成！</div>', unsafe_allow_html=True)

    st.balloons()

    st.write("### 处理摘要")

    # 显示所有步骤的状态
    for step in range(5):
        if step in st.session_state.workflow_state['approved_steps']:
            st.success(f"✅ Step {step}: 已完成并批准")
        elif step in st.session_state.workflow_state['step_results']:
            st.info(f"ℹ️ Step {step}: 已运行但未批准")
        else:
            st.text(f"⬜ Step {step}: 未运行")

    # 显示最终参数
    with st.expander("📋 使用的参数"):
        params = st.session_state.workflow_state
        st.json({
            "instrument": params['instrument'],
            "step1": {
                "detect_thresh": params['detect_thresh'],
                "min_area": params['min_area']
            },
            "step2": {
                "spike_width": params['spike_width'],
                "spike_length": params['spike_length'],
                "rotation_angle": params['rotation_angle']
            },
            "step3": {
                "dilation_factor": params['dilation_factor'],
                "kernel_shape": params['kernel_shape']
            },
            "step4": {
                "interpolation_method": params['interpolation_method']
            }
        })

    # 重新开始
    if st.button("🔄 开始新的处理", key='restart'):
        st.session_state.workflow_state = {
            'current_step': 0,
            'image_path': None,
            'instrument': 'HST',
            'step_results': {},
            'approved_steps': set(),
            'detect_thresh': 1.5,
            'min_area': 5,
            'excluded_regions': [],
            'spike_width': 3.0,
            'spike_length': 150.0,
            'rotation_angle': 0.0,
            'dilation_factor': 1.5,
            'kernel_shape': 'disk',
            'interpolation_method': 'rbf',
        }
        st.rerun()


def main():
    """主函数"""
    st.markdown('<div class="main-header">🔭 ICL Workflow - 人在回路系统</div>',
                unsafe_allow_html=True)

    st.write("""
    这是一个用于强透镜星系团ICL（星系团内光）剥离的交互式工作流系统。

    **工作流程:**
    1. **Step 0**: 上传FITS图像，评估场复杂度
    2. **Step 1**: 提取初始星系Mask，可调检测阈值
    3. **Step 2**: 生成衍射星芒Mask，可调几何参数
    4. **Step 3**: 合并并膨胀Mask，可调膨胀系数
    5. **Step 4**: ICL插值与剥离，可调插值方法
    6. **Step 5**: 质量评估与结果导出
    """)

    # 初始化
    init_session_state()

    # 侧边栏导航
    with st.sidebar:
        st.subheader("📍 导航")
        current = st.session_state.workflow_state['current_step']
        steps = ["0: 上传", "1: 初始Mask", "2: 星芒Mask", "3: 合并膨胀", "4: ICL剥离", "5: 完成"]

        for i, step_name in enumerate(steps):
            if i == current:
                st.markdown(f"**→ {step_name}**")
            elif i in st.session_state.workflow_state['approved_steps']:
                st.markdown(f"✅ {step_name}")
            else:
                st.text(f"  {step_name}")

    # 根据当前步骤显示对应界面
    current_step = st.session_state.workflow_state['current_step']

    if current_step == 0:
        step_0_file_upload()
    elif current_step == 1:
        step_1_initial_mask()
    elif current_step == 2:
        step_2_spike_mask()
    elif current_step == 3:
        step_3_merge_dilate()
    elif current_step == 4:
        step_4_interpolation()
    elif current_step == 5:
        step_5_summary()


if __name__ == "__main__":
    main()
