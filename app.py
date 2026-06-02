#!/usr/bin/env python3
"""
ICL Workflow - Human-in-the-Loop Visual System
基于Streamlit的可视化人机交互系统
新增：右侧日志面板
"""

import streamlit as st
import sys
from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.visualization import simple_norm
import matplotlib.pyplot as plt
import io
import logging
import queue
import threading
from datetime import datetime

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


# ========== 日志系统 ==========
class StreamlitLogHandler(logging.Handler):
    """自定义日志处理器，将日志发送到Streamlit"""
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue
        self.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        ))

    def emit(self, record):
        msg = self.format(record)
        self.log_queue.put(msg)


class LogCapture:
    """日志捕获器"""
    def __init__(self):
        self.log_queue = queue.Queue()
        self.handler = StreamlitLogHandler(self.log_queue)
        self.handler.setLevel(logging.INFO)

        # 配置根日志记录器
        self.root_logger = logging.getLogger()
        self.root_logger.addHandler(self.handler)

        # 也配置ICLWorkflow日志
        self.workflow_logger = logging.getLogger("ICLWorkflow")
        self.workflow_logger.setLevel(logging.INFO)

    def get_logs(self, clear=True):
        """获取所有日志"""
        logs = []
        while not self.log_queue.empty():
            try:
                logs.append(self.log_queue.get_nowait())
            except queue.Empty:
                break
        return logs

    def clear(self):
        """清空日志队列"""
        while not self.log_queue.empty():
            try:
                self.log_queue.get_nowait()
            except queue.Empty:
                break


# ========== 页面配置 ==========
st.set_page_config(
    page_title="ICL Workflow - HITL",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== CSS样式 ==========
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
    .log-container {
        background-color: #1e1e1e;
        color: #d4d4d4;
        font-family: 'Courier New', monospace;
        font-size: 11px;
        padding: 10px;
        border-radius: 5px;
        height: calc(100vh - 200px);
        overflow-y: auto;
        white-space: pre-wrap;
        word-wrap: break-word;
    }
    .log-info { color: #4ec9b0; }
    .log-warning { color: #dcdcaa; }
    .log-error { color: #f44747; }
    .log-debug { color: #808080; }
    .log-header {
        background-color: #333;
        color: #fff;
        padding: 5px 10px;
        border-radius: 5px 5px 0 0;
        font-weight: bold;
    }
    .stColumn {
        padding: 0 5px;
    }
</style>
""", unsafe_allow_html=True)


# ========== 初始化 ==========
def init_session_state():
    """初始化会话状态"""
    if 'workflow_state' not in st.session_state:
        st.session_state.workflow_state = {
            'current_step': 0,
            'image_path': None,
            'instrument': 'HST',
            'step_results': {},
            'approved_steps': set(),
            'detect_thresh': 3.0,  # JWST建议更高阈值
            'min_area': 10,
            'excluded_regions': [],
            'center_x': None,
            'center_y': None,
            'spike_width': 5.0,  # JWST PSF更大
            'spike_length': 200.0,
            'rotation_angle': 0.0,
            'dilation_factor': 1.3,  # JWST建议较小膨胀
            'kernel_shape': 'disk',
            'interpolation_method': 'linear',  # JWST大图建议linear
            'logs': [],  # 存储日志
        }

    if 'log_capture' not in st.session_state:
        st.session_state.log_capture = LogCapture()


def add_log(message, level="INFO"):
    """添加日志到会话状态"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] [{level}] {message}"
    st.session_state.workflow_state['logs'].append(log_entry)
    # 限制日志数量
    if len(st.session_state.workflow_state['logs']) > 1000:
        st.session_state.workflow_state['logs'] = st.session_state.workflow_state['logs'][-500:]


def render_logs():
    """渲染日志面板"""
    st.markdown("### 📋 系统日志")

    # 日志控制按钮
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("🔄 刷新", key="refresh_logs"):
            pass  # 自动刷新
    with col2:
        if st.button("🗑️ 清空", key="clear_logs"):
            st.session_state.workflow_state['logs'] = []
            st.rerun()
    with col3:
        if st.button("💾 导出", key="export_logs"):
            logs_text = "\n".join(st.session_state.workflow_state['logs'])
            st.download_button(
                "下载日志",
                logs_text,
                file_name=f"icl_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )

    # 日志级别过滤
    log_level = st.selectbox(
        "日志级别",
        ["ALL", "INFO", "WARNING", "ERROR", "DEBUG"],
        index=0,
        key="log_level_filter"
    )

    # 显示日志
    logs = st.session_state.workflow_state['logs']

    # 应用过滤
    if log_level != "ALL":
        logs = [log for log in logs if f"[{log_level}]" in log]

    # 构建HTML显示
    log_html = '<div class="log-container">'

    if not logs:
        log_html += '<span class="log-debug">暂无日志...</span>'
    else:
        for log in logs[-200:]:  # 只显示最后200条
            # 根据日志级别着色
            if "ERROR" in log:
                log_html += f'<div class="log-error">{log}</div>'
            elif "WARNING" in log:
                log_html += f'<div class="log-warning">{log}</div>'
            elif "DEBUG" in log:
                log_html += f'<div class="log-debug">{log}</div>'
            else:
                log_html += f'<div class="log-info">{log}</div>'

    log_html += '</div>'

    st.markdown(log_html, unsafe_allow_html=True)


# ========== 可视化函数 ==========
def display_fits_image(data, title, colormap='gray', log_scale=True):
    """显示FITS图像"""
    fig, ax = plt.subplots(figsize=(10, 10))

    if log_scale and np.all(data > 0):
        norm = simple_norm(data, 'log', percent=99)
    else:
        norm = simple_norm(data, 'linear', percent=99)

    im = ax.imshow(data, norm=norm, cmap=colormap, origin='lower')
    ax.set_title(title, fontsize=12)
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.axis('off')

    return fig


# ========== 步骤函数 ==========
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
        add_log(f"文件上传成功: {uploaded_file.name}")

        # 读取并显示预览
        try:
            with fits.open(save_path) as hdul:
                data = hdul[0].data
                if data is None and len(hdul) > 1:
                    data = hdul[1].data

                st.session_state.workflow_state['image_shape'] = data.shape
                add_log(f"图像尺寸: {data.shape}, 数据类型: {data.dtype}")

                # 显示图像
                fig = display_fits_image(data, "Input Image Preview")
                st.pyplot(fig)

                # 图像信息
                st.write("**图像信息**")
                st.write(f"尺寸: {data.shape}")
                st.write(f"数值范围: [{data.min():.2e}, {data.max():.2e}]")

                # 显示FITS头信息
                with st.expander("查看FITS头信息"):
                    st.text(str(hdul[0].header))

                # 选择仪器
                instrument = st.selectbox(
                    "选择望远镜/仪器",
                    ['JWST', 'HST', 'CSST', 'Euclid'],
                    index=0,  # JWST默认
                    key='instrument_select'
                )
                st.session_state.workflow_state['instrument'] = instrument
                add_log(f"选择仪器: {instrument}")

                # JWST提示
                if instrument == 'JWST':
                    st.info("💡 JWST数据提示：\n- 建议使用检测阈值 ≥ 3.0\n- 建议使用 'linear' 插值方法\n- 建议使用较小的膨胀系数 (1.2-1.5)")

        except Exception as e:
            add_log(f"读取图像失败: {str(e)}", "ERROR")
            st.error(f"读取图像失败: {str(e)}")
            return

        # 场复杂度评估
        if st.button("评估场复杂度", key='eval_complexity'):
            add_log("开始场复杂度评估...")
            with st.spinner("分析中..."):
                try:
                    result = ToolEvaluateFieldComplexity.execute(
                        str(save_path),
                        output_dir="outputs/app"
                    )

                    st.session_state.workflow_state['step_results'][0] = result

                    if not result.success or result.metrics is None:
                        error_msg = result.message
                        add_log(f"场复杂度评估失败: {error_msg}", "ERROR")
                        st.error(f"场复杂度评估失败: {error_msg}")
                        st.info("提示: 对于JWST数据，尝试在左侧提高'检测阈值'到 3.0 或更高")
                    else:
                        metrics = result.metrics
                        add_log(f"评估成功: 复杂度={metrics.get('complexity_score', 0):.1f}, "
                               f"亮星={metrics.get('num_bright_stars', 0)}, "
                               f"暗星={metrics.get('num_faint_stars', 0)}")

                        # 显示结果
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        cols = st.columns(4)
                        with cols[0]:
                            st.metric("复杂度评分", f"{metrics.get('complexity_score', 0):.1f}")
                        with cols[1]:
                            st.metric("亮星数量", int(metrics.get('num_bright_stars', 0)))
                        with cols[2]:
                            st.metric("暗星数量", int(metrics.get('num_faint_stars', 0)))
                        with cols[3]:
                            mode = "AUTO" if metrics.get('recommended_auto_mode', False) else "MANUAL"
                            st.metric("推荐模式", mode)
                        st.markdown('</div>', unsafe_allow_html=True)

                        st.info(result.message)

                        # 下一步按钮
                        if st.button("进入 Step 1 →", key='to_step1'):
                            st.session_state.workflow_state['current_step'] = 1
                            add_log("进入 Step 1")
                            st.rerun()

                except Exception as e:
                    add_log(f"评估过程异常: {str(e)}", "ERROR")
                    st.error(f"评估过程异常: {str(e)}")


def step_1_initial_mask():
    """Step 1: 初始Mask提取"""
    st.markdown('<div class="step-header">Step 1: 初始星系Mask提取</div>', unsafe_allow_html=True)

    image_path = st.session_state.workflow_state['image_path']
    instrument = st.session_state.workflow_state.get('instrument', 'HST')

    # 根据仪器给出建议
    if instrument == 'JWST':
        st.info("💡 JWST建议: 检测阈值 ≥ 3.0, 最小面积 ≥ 10")

    # 参数控制
    col_param, col_run = st.columns([2, 1])

    with col_param:
        detect_thresh = st.slider(
            "检测阈值 (sigma)",
            min_value=0.5, max_value=10.0,
            value=st.session_state.workflow_state['detect_thresh'],
            step=0.5,
            help="JWST建议使用 3.0-5.0"
        )

        min_area = st.slider(
            "最小像素面积",
            min_value=1, max_value=100,
            value=st.session_state.workflow_state['min_area'],
            step=5,
            help="JWST建议使用 10-20"
        )

    with col_run:
        st.write("")
        st.write("")
        run_button = st.button("🚀 运行提取", key='run_step1', use_container_width=True)

    st.session_state.workflow_state['detect_thresh'] = detect_thresh
    st.session_state.workflow_state['min_area'] = min_area

    if run_button:
        add_log(f"Step 1: 开始提取 (阈值={detect_thresh}, 最小面积={min_area})")

        with st.spinner("运行SEP源提取..."):
            try:
                result = ToolExtractInitialMask.execute(
                    image_path,
                    detect_thresh=detect_thresh,
                    min_area=min_area,
                    output_dir="outputs/app",
                    check_region_size=100
                )

                st.session_state.workflow_state['step_results'][1] = result

                if result.success:
                    add_log(f"提取成功: 检测到 {result.metrics.get('num_objects', 0)} 个天体")

                    # 读取结果
                    mask = fits.getdata(result.output_path)
                    original = fits.getdata(image_path)

                    # 显示指标
                    cols = st.columns(4)
                    with cols[0]:
                        st.metric("检测天体数", result.metrics.get('num_objects', 0))
                    with cols[1]:
                        st.metric("中心X", f"{result.metrics.get('center_x', 0):.1f}")
                    with cols[2]:
                        st.metric("中心Y", f"{result.metrics.get('center_y', 0):.1f}")
                    with cols[3]:
                        if result.metrics.get('arc_warning'):
                            st.error("⚠️ 可能存在弧污染")
                            add_log("警告: 可能存在弧污染", "WARNING")
                        else:
                            st.success("✓ 无明显弧污染")

                    # 保存中心坐标
                    st.session_state.workflow_state['center_x'] = result.metrics.get('center_x')
                    st.session_state.workflow_state['center_y'] = result.metrics.get('center_y')

                    # 显示图像
                    tab1, tab2 = st.tabs(["原图+Mask", "可视化报告"])

                    with tab1:
                        fig, ax = plt.subplots(figsize=(10, 10))
                        norm = simple_norm(original, 'log', percent=99)
                        ax.imshow(original, norm=norm, cmap='gray', origin='lower')
                        ax.contour(mask, levels=[0.5], colors='red', linewidths=1.5, alpha=0.7)
                        ax.set_title("Original with Mask Overlay")
                        ax.axis('off')
                        st.pyplot(fig)

                    with tab2:
                        if result.visualization_path and Path(result.visualization_path).exists():
                            st.image(result.visualization_path)

                    # 人工审核
                    st.markdown("---")
                    st.subheader("人工审核")

                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("✅ 批准 - 进入下一步", key='approve_step1', use_container_width=True):
                            st.session_state.workflow_state['approved_steps'].add(1)
                            st.session_state.workflow_state['current_step'] = 2
                            add_log("Step 1: 已批准，进入 Step 2")
                            st.rerun()

                    with col2:
                        if st.button("❌ 拒绝 - 调整参数", key='reject_step1', use_container_width=True):
                            add_log("Step 1: 被拒绝，调整参数")
                            st.warning("请在上方调整参数后重新运行")

                else:
                    error_msg = result.message
                    add_log(f"提取失败: {error_msg}", "ERROR")
                    st.error(f"提取失败: {error_msg}")

                    if "pixstack" in error_msg.lower():
                        st.info("提示: 像素缓冲区溢出。请提高'检测阈值'到 5.0 或更高，或增大'最小面积'")

            except Exception as e:
                add_log(f"提取过程异常: {str(e)}", "ERROR")
                st.error(f"提取过程异常: {str(e)}")


def step_2_spike_mask():
    """Step 2: 衍射星芒Mask"""
    st.markdown('<div class="step-header">Step 2: 衍射星芒Mask生成</div>', unsafe_allow_html=True)

    image_path = st.session_state.workflow_state['image_path']
    instrument = st.session_state.workflow_state['instrument']
    cx = st.session_state.workflow_state.get('center_x')
    cy = st.session_state.workflow_state.get('center_y')

    if cx is None or cy is None:
        st.error("请先完成Step 1以获取中心坐标")
        return

    # 参数控制
    st.write(f"**透镜星系中心**: ({cx:.1f}, {cy:.1f})")
    st.write(f"**仪器**: {instrument}")

    if instrument == 'JWST':
        st.info("💡 JWST的NIRCam有6条衍射星芒，PSF比HST更大")

    col_params = st.columns(3)

    with col_params[0]:
        spike_width = st.slider("星芒宽度", 1.0, 15.0,
                               st.session_state.workflow_state.get('spike_width', 5.0), 0.5)
    with col_params[1]:
        spike_length = st.slider("星芒长度", 50.0, 400.0,
                                st.session_state.workflow_state.get('spike_length', 200.0), 10.0)
    with col_params[2]:
        rotation_angle = st.slider("旋转角度", -45.0, 45.0,
                                  st.session_state.workflow_state.get('rotation_angle', 0.0), 1.0)

    st.session_state.workflow_state.update({
        'spike_width': spike_width,
        'spike_length': spike_length,
        'rotation_angle': rotation_angle
    })

    if st.button("🚀 生成星芒Mask", key='run_step2'):
        add_log(f"Step 2: 生成星芒Mask (宽度={spike_width}, 长度={spike_length})")

        with st.spinner("生成星芒Mask..."):
            try:
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
                    add_log(f"星芒Mask生成成功: {result.metrics.get('num_spikes', 0)}条星芒")

                    # 显示结果
                    spike_mask = fits.getdata(result.output_path)
                    original = fits.getdata(image_path)

                    cols = st.columns(4)
                    with cols[0]:
                        st.metric("星芒数量", result.metrics.get('num_spikes', 4))
                    with cols[1]:
                        st.metric("宽度", f"{result.metrics.get('spike_width', 0):.1f}pix")
                    with cols[2]:
                        st.metric("长度", f"{result.metrics.get('spike_length', 0):.1f}pix")
                    with cols[3]:
                        st.metric("旋转角", f"{result.metrics.get('rotation_angle', 0):.1f}°")

                    # 显示叠加图
                    fig, ax = plt.subplots(figsize=(10, 10))
                    norm = simple_norm(original, 'log', percent=99)
                    ax.imshow(original, norm=norm, cmap='gray', origin='lower')
                    ax.contour(spike_mask, levels=[0.5], colors='lime', linewidths=2, alpha=0.7)
                    ax.plot(cx, cy, 'r+', markersize=20, markeredgewidth=2)
                    ax.set_title("Spike Mask Overlay")
                    ax.axis('off')
                    st.pyplot(fig)

                    # 审核
                    st.markdown("---")
                    st.subheader("人工审核")

                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("✅ 批准", key='approve_step2', use_container_width=True):
                            st.session_state.workflow_state['approved_steps'].add(2)
                            st.session_state.workflow_state['current_step'] = 3
                            add_log("Step 2: 已批准")
                            st.rerun()
                    with col2:
                        if st.button("❌ 拒绝", key='reject_step2', use_container_width=True):
                            add_log("Step 2: 被拒绝")
                            st.warning("请调整参数后重新运行")

                else:
                    add_log(f"星芒Mask生成失败: {result.message}", "ERROR")
                    st.error(f"生成失败: {result.message}")

            except Exception as e:
                add_log(f"生成过程异常: {str(e)}", "ERROR")
                st.error(f"生成过程异常: {str(e)}")


def step_3_merge_dilate():
    """Step 3: Mask合并与膨胀"""
    st.markdown('<div class="step-header">Step 3: Mask合并与膨胀</div>', unsafe_allow_html=True)

    image_path = st.session_state.workflow_state['image_path']
    step1_result = st.session_state.workflow_state['step_results'].get(1)
    step2_result = st.session_state.workflow_state['step_results'].get(2)

    if not step1_result or not step2_result:
        st.error("请先完成Step 1和Step 2")
        return

    # 参数
    col1, col2 = st.columns(2)
    with col1:
        dilation_factor = st.slider("膨胀系数", 0.5, 3.0,
                                   st.session_state.workflow_state['dilation_factor'], 0.1,
                                   help="JWST建议1.2-1.5，HST建议1.5")
    with col2:
        kernel_shape = st.selectbox("核形状", ['disk', 'box'],
                                   index=0 if st.session_state.workflow_state['kernel_shape'] == 'disk' else 1)

    st.session_state.workflow_state.update({
        'dilation_factor': dilation_factor,
        'kernel_shape': kernel_shape
    })

    if st.button("🚀 合并与膨胀"):
        add_log(f"Step 3: 合并Mask (膨胀系数={dilation_factor})")

        with st.spinner("处理中..."):
            try:
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
                    orig_cov = result.metrics.get('original_coverage', 0)
                    dil_cov = result.metrics.get('dilated_coverage', 0)
                    add_log(f"合并成功: 覆盖率 {orig_cov:.1f}% -> {dil_cov:.1f}%")

                    cols = st.columns(3)
                    with cols[0]:
                        st.metric("原始覆盖率", f"{orig_cov:.1f}%")
                    with cols[1]:
                        st.metric("膨胀后覆盖率", f"{dil_cov:.1f}%")
                    with cols[2]:
                        increase = dil_cov - orig_cov
                        st.metric("增加", f"{increase:.1f}%")

                    if dil_cov > 30:
                        st.warning("⚠️ 覆盖率过高，可能吞没背景源")
                        add_log("警告: 覆盖率过高", "WARNING")

                    if result.visualization_path and Path(result.visualization_path).exists():
                        st.image(result.visualization_path)

                    # 审核
                    st.markdown("---")
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("✅ 批准", key='approve_step3'):
                            st.session_state.workflow_state['approved_steps'].add(3)
                            st.session_state.workflow_state['current_step'] = 4
                            add_log("Step 3: 已批准")
                            st.rerun()
                    with col2:
                        if st.button("❌ 拒绝", key='reject_step3'):
                            add_log("Step 3: 被拒绝")

                else:
                    add_log(f"合并失败: {result.message}", "ERROR")
                    st.error(f"合并失败: {result.message}")

            except Exception as e:
                add_log(f"合并过程异常: {str(e)}", "ERROR")
                st.error(f"合并过程异常: {str(e)}")


def step_4_interpolation():
    """Step 4: ICL插值与剥离"""
    st.markdown('<div class="step-header">Step 4: ICL插值与剥离</div>', unsafe_allow_html=True)

    image_path = st.session_state.workflow_state['image_path']
    step3_result = st.session_state.workflow_state['step_results'].get(3)

    if not step3_result:
        st.error("请先完成Step 3")
        return

    # 参数
    method = st.selectbox(
        "插值方法",
        ['linear', 'rbf', 'clough-tocher', 'nearest'],
        index=['linear', 'rbf', 'clough-tocher', 'nearest'].index(
            st.session_state.workflow_state['interpolation_method']
        ),
        help="linear:快速, rbf:质量最高但慢"
    )

    st.session_state.workflow_state['interpolation_method'] = method

    if st.button("🚀 运行ICL剥离"):
        add_log(f"Step 4: 开始ICL剥离 (方法={method})")

        with st.spinner("运行中（可能需要几分钟）..."):
            try:
                result = ToolInterpolateAndSubtract.execute(
                    image_path,
                    step3_result.output_path,
                    method=method,
                    output_dir="outputs/app"
                )

                st.session_state.workflow_state['step_results'][4] = result

                if result.success:
                    metrics = result.metrics
                    add_log(f"剥离完成: 综合评分={metrics.get('overall_score', 0):.1f}/100, "
                           f"状态={'通过' if metrics.get('passed') else '未通过'}")

                    # 显示质量指标
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    cols = st.columns(4)
                    with cols[0]:
                        st.metric("综合评分", f"{metrics.get('overall_score', 0):.1f}/100")
                    with cols[1]:
                        st.metric("通量守恒", f"{metrics.get('flux_conservation_score', 0):.1f}/100")
                    with cols[2]:
                        st.metric("形态保持", f"{metrics.get('shape_preservation_score', 0):.1f}/100")
                    with cols[3]:
                        status = "✅ 通过" if metrics.get('passed') else "❌ 未通过"
                        st.metric("状态", status)
                    st.markdown('</div>', unsafe_allow_html=True)

                    # 详细指标
                    with st.expander("📊 详细质量指标"):
                        st.json({k: v for k, v in metrics.items() if isinstance(v, (int, float, bool, str))})

                    if result.visualization_path and Path(result.visualization_path).exists():
                        st.image(result.visualization_path, use_column_width=True)

                    # 最终审核
                    st.markdown("---")
                    if metrics.get('passed'):
                        st.success("✅ 质量检验通过！结果可用于科学分析。")
                    else:
                        st.error("❌ 质量检验未通过，建议调整参数重试")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button("✅ 完成工作流", key='approve_step4'):
                            st.session_state.workflow_state['approved_steps'].add(4)
                            st.session_state.workflow_state['current_step'] = 5
                            add_log("工作流完成！")
                            st.rerun()
                    with col2:
                        if st.button("🔙 返回调整"):
                            st.session_state.workflow_state['current_step'] = 3
                            add_log("返回 Step 3 调整")
                            st.rerun()
                    with col3:
                        if result.output_path and Path(result.output_path).exists():
                            with open(result.output_path, 'rb') as f:
                                st.download_button("📥 下载残差图", f,
                                                 file_name="clean_residual.fits")

                else:
                    add_log(f"剥离失败: {result.message}", "ERROR")
                    st.error(f"剥离失败: {result.message}")

            except Exception as e:
                add_log(f"剥离过程异常: {str(e)}", "ERROR")
                st.error(f"剥离过程异常: {str(e)}")


def step_5_summary():
    """Step 5: 工作流总结"""
    st.markdown('<div class="step-header">工作流完成！</div>', unsafe_allow_html=True)

    st.balloons()

    st.write("### 处理摘要")

    for step in range(5):
        if step in st.session_state.workflow_state['approved_steps']:
            st.success(f"✅ Step {step}: 已完成并批准")
        elif step in st.session_state.workflow_state['step_results']:
            st.info(f"ℹ️ Step {step}: 已运行但未批准")
        else:
            st.text(f"⬜ Step {step}: 未运行")

    # 参数摘要
    with st.expander("📋 使用的参数"):
        params = st.session_state.workflow_state
        st.json({
            "instrument": params.get('instrument', 'HST'),
            "step1": {"detect_thresh": params.get('detect_thresh'), "min_area": params.get('min_area')},
            "step2": {"spike_width": params.get('spike_width'), "spike_length": params.get('spike_length')},
            "step3": {"dilation_factor": params.get('dilation_factor'), "kernel_shape": params.get('kernel_shape')},
            "step4": {"interpolation_method": params.get('interpolation_method')}
        })

    if st.button("🔄 开始新的处理"):
        # 保留日志但重置其他状态
        logs = st.session_state.workflow_state.get('logs', [])
        st.session_state.workflow_state = {
            'current_step': 0,
            'image_path': None,
            'instrument': 'JWST',
            'step_results': {},
            'approved_steps': set(),
            'detect_thresh': 3.0,
            'min_area': 10,
            'excluded_regions': [],
            'spike_width': 5.0,
            'spike_length': 200.0,
            'rotation_angle': 0.0,
            'dilation_factor': 1.3,
            'kernel_shape': 'disk',
            'interpolation_method': 'linear',
            'logs': logs,  # 保留日志
        }
        add_log("=" * 50)
        add_log("开始新的处理")
        st.rerun()


# ========== 主函数 ==========
def main():
    """主函数 - 3列布局"""
    st.markdown('<div class="main-header">🔭 ICL Workflow - 人在回路系统</div>',
                unsafe_allow_html=True)

    init_session_state()

    # ========== 3列布局 ==========
    # 列1: 侧边栏参数（通过st.sidebar）
    # 列2: 主内容
    # 列3: 日志面板

    # 侧边栏 - 导航和全局信息
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

        st.markdown("---")
        st.write("**当前参数**")
        st.write(f"仪器: {st.session_state.workflow_state.get('instrument', 'HST')}")
        st.write(f"阈值: {st.session_state.workflow_state.get('detect_thresh', 1.5)}")
        st.write(f"插值: {st.session_state.workflow_state.get('interpolation_method', 'rbf')}")

    # 主区域 - 2列: 内容和日志
    col_main, col_logs = st.columns([3, 1])

    with col_main:
        # 显示当前步骤
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

    with col_logs:
        # 日志面板
        render_logs()


if __name__ == "__main__":
    main()
