# Chainxim-PouwPrism

中文 | [English](README.md)

**ChainXim** 是由 XinLab 开发的高性能区块链仿真器。本项目基于 ChainXim 开发，集成了 **Prism (并行链架构)** 与 **有效工作量证明 (Proof of Useful Work, PoUW)** 共识机制。

**项目地址:** [https://github.com/Candytu1t/chainxim_pouw](https://github.com/Candytu1t/chainxim_pouw)

## 📖 简介 Introduction

本项目是基于 **ChainXim** 框架开发的区块链仿真器，重点针对 **Prism (并行链架构)** 与 **有效工作量证明 (PoUW)** 共识机制进行了定制与集成。

### 核心特性

*   **Prism 架构**: 实现了并行链结构，将交易块、提案块和投票块解耦，以最大化吞吐量并最小化延迟。
*   **PoUW 共识**: 使用实际的优化问题替代无用的哈希计算。仿真器支持两种不同的优化判定标准：
    *   **KKT 方法**: 求解优化问题直到满足 **Karush-Kuhn-Tucker (KKT)** 条件，确保获得高质量的解。
    *   **阈值 (Threshold) 方法**: 接受达到特定改进阈值的解，优先考虑速度和收敛性。

去中心化系统的模拟往往面临巨大的成本与工程挑战。本仿真器提供了一种低成本、大规模的解决方案，支持在单机上部署并模拟大规模区块链网络。它能够在虚拟环境中模拟节点的运作与交互，并在主链达到预定高度后自动生成吞吐量、分叉率、链质量等关键性能指标报告。


## 🚀 快速开始 Quick Start

### 前置要求
*   **Anaconda**: [下载链接](https://www.anaconda.com/download)
*   **Python**: 3.10

### 安装步骤

1.  **克隆仓库:**
    ```bash
    git clone https://github.com/Candytu1t/chainxim_pouw.git
    cd chainxim_pouw
    ```

2.  **创建并激活环境:**
    ```bash
    conda create -n chainxim python=3.10 python-graphviz
    conda activate chainxim
    ```

3.  **安装依赖:**
    ```bash
    pip install -r requirements.txt
    ```

### 运行仿真

使用 PoUW 配置运行仿真器：

```bash
python main.py --config pouw_config.ini
```



## 🔗 相关项目 Related Projects

*   **Chainxim**: A blockchain simulator developed by XinLab. [源代码](https://github.com/ChainXim-Team/ChainXim)

