# PWOptimization

![Python](https://img.shields.io/badge/Python-3.10-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-0.24-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-1.13-red)
![NumPy](https://img.shields.io/badge/NumPy-1.24-blue)
![Pandas](https://img.shields.io/badge/Pandas-1.5-lightgrey)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7-purple)
![License](https://img.shields.io/badge/license-MIT-green)

Comparison between **Frank–Wolfe (FW)** and **Sequential Minimal Optimization (SMO, MVP variant)** methods for solving the **SVM dual problem**.

## Requirements

* **Conda** installed (Anaconda or Miniconda).
* **`optimization.yml`** file containing the project dependencies.

## Reproducing the Results

### 1) Create the Conda environment `opt`
```bash
conda env create -f optimization.yml
conda activate opt
````

### 2) Run the experiments

Execute:

```bash
python optimization.py
```

> Outputs (accuracies, timings, metrics) will be printed/saved according to the script configuration.

## Graph Visualization

Due to graphical rendering issues on the University machines, the plots were generated using **Google Colab** via the `plot_methods_comparison` function.

Colab notebook for graph visualization:
👉 [Open in Google Colab](https://colab.research.google.com/drive/1s8AJa46PFZQc8DH2OVR-neCOXgTnVibg)
