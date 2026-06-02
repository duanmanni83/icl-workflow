#!/usr/bin/env python3
"""简化版 - 用于调试"""
import streamlit as st
from pathlib import Path
import sys
sys.path.insert(0, 'src')

st.set_page_config(page_title="ICL Simple", layout="wide")

# 初始化
if 'step' not in st.session_state:
    st.session_state.step = 0
    st.session_state.has_file = False

st.title("🔭 ICL Workflow - 简化版")

# 侧边栏显示当前状态
with st.sidebar:
    st.write(f"当前步骤: {st.session_state.step}")
    st.write(f"有文件: {st.session_state.has_file}")

# 主内容
if st.session_state.step == 0:
    st.header("Step 0: 上传文件")

    uploaded = st.file_uploader("选择FITS文件", type=['fits'])

    if uploaded:
        st.session_state.has_file = True
        st.success(f"已上传: {uploaded.name}")

        if st.button("→ 进入 Step 1"):
            st.session_state.step = 1
            st.rerun()

elif st.session_state.step == 1:
    st.header("Step 1: 初始Mask")

    if not st.session_state.has_file:
        st.error("请先上传文件")
        if st.button("← 返回 Step 0"):
            st.session_state.step = 0
            st.rerun()
    else:
        st.success("成功进入 Step 1!")

        # 推荐值按钮
        instrument = st.selectbox("仪器", ['JWST', 'HST'])

        if instrument == 'JWST':
            rec_thresh, rec_area = 3.0, 10
            st.info(f"JWST推荐: 阈值={rec_thresh}, 面积={rec_area}")
        else:
            rec_thresh, rec_area = 1.5, 5
            st.info(f"HST推荐: 阈值={rec_thresh}, 面积={rec_area}")

        if st.button("🎯 使用推荐值"):
            st.success(f"已设置: 阈值={rec_thresh}, 面积={rec_area}")

        thresh = st.slider("阈值", 0.5, 10.0, rec_thresh)
        area = st.slider("面积", 1, 100, rec_area)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ 批准 - 去Step 2"):
                st.session_state.step = 2
                st.rerun()
        with col2:
            if st.button("← 返回 Step 0"):
                st.session_state.step = 0
                st.rerun()

elif st.session_state.step == 2:
    st.header("Step 2: 星芒Mask")
    st.success("成功进入 Step 2!")

    if st.button("← 返回 Step 1"):
        st.session_state.step = 1
        st.rerun()

st.markdown("---")
st.caption("如果此简化版能正常工作，说明主应用有某个特定问题")
