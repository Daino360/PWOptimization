# PWOptimization

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
