# Chainxim-PouwPrism

[中文](README.zh.md) | English

**ChainXim** is a high-performance blockchain simulator developed by XinLab. This project is developed based on ChainXim, integrated with **Prism (Parallel Chain Architecture)** and **Proof of Useful Work (PoUW)** consensus.

**Repository:** [https://github.com/Candytu1t/chainxim_pouw](https://github.com/Candytu1t/chainxim_pouw)

## 📖 Introduction

This project is a specialized blockchain simulator built upon the **ChainXim** framework, customized for research on **Prism**—a high-throughput parallel chain architecture—and **Proof of Useful Work (PoUW)** consensus.

### Key Features

*   **Prism Architecture**: Implements a parallel chain structure where transaction blocks, proposer blocks, and voter blocks are decoupled to maximize throughput and minimize latency.
*   **PoUW Consensus**: Replaces useless hash calculations with practical optimization problems. The simulator supports two distinct optimization criteria:
    *   **KKT Method**: Solves optimization problems until **Karush-Kuhn-Tucker (KKT)** conditions are met, ensuring high-quality solutions.
    *   **Threshold Method**: Accepts solutions that achieve a specific improvement threshold, prioritizing speed and convergence.

Simulating large-scale decentralized networks often presents significant costs and engineering challenges. This simulator addresses these issues by enabling low-cost, large-scale blockchain deployment and testing on a single machine. It simulates node operations and interactions within a virtual environment, generating comprehensive performance reports (throughput, fork rate, chain quality) once the main chain reaches a target height.

![intro](doc/intro.svg)

## 🚀 Quick Start

### Prerequisites
*   **Anaconda**: [Download](https://www.anaconda.com/download)
*   **Python**: 3.10

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Candytu1t/chainxim_pouw.git
    cd chainxim_pouw
    ```

2.  **Create and activate environment:**
    ```bash
    conda create -n chainxim python=3.10 python-graphviz
    conda activate chainxim
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Simulator

To run the simulator with the Proof of Useful Work (PoUW) configuration:

```bash
python main.py --config pouw_config.ini
```


##  Related Projects

*   **Chainxim**: A blockchain simulator developed by XinLab. [Source Code](https://github.com/ChainXim-Team/ChainXim)