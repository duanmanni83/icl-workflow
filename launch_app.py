#!/usr/bin/env python3
"""
Launch the ICL Workflow HITL Application
启动人在回路可视化系统
"""

import subprocess
import sys
from pathlib import Path


def check_dependencies():
    """检查必要的依赖"""
    try:
        import streamlit
        print("✓ Streamlit 已安装")
    except ImportError:
        print("✗ Streamlit 未安装，正在安装...")
        subprocess.run([sys.executable, "-m", "pip", "install", "streamlit", "-q"])
        print("✓ Streamlit 安装完成")

    try:
        import astropy
        print("✓ Astropy 已安装")
    except ImportError:
        print("✗ Astropy 未安装，正在安装...")
        subprocess.run([sys.executable, "-m", "pip", "install", "astropy", "-q"])


def main():
    """启动应用"""
    print("=" * 60)
    print("ICL Workflow - Human-in-the-Loop System")
    print("人在回路可视化系统")
    print("=" * 60)

    # 检查依赖
    print("\n检查依赖...")
    check_dependencies()

    # 确保目录存在
    Path("uploads").mkdir(exist_ok=True)
    Path("outputs/app").mkdir(parents=True, exist_ok=True)

    print("\n启动 Streamlit 应用...")
    print("应用将在浏览器中打开")
    print("-" * 60)

    # 启动streamlit
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        "app.py",
        "--server.port", "8501",
        "--server.address", "localhost",
        "--browser.serverAddress", "localhost"
    ])


if __name__ == "__main__":
    main()
