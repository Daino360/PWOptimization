# PWOptimization

Confronto tra i metodi **Frank–Wolfe (FW)** e **Sequential Minimal Optimization (SMO, variante MVP)** per la risoluzione del **problema duale SVM**.

## Requisiti

* **Conda** installato (Anaconda o Miniconda).
* File **`optimization.yaml`** con le dipendenze del progetto.

## Riproduzione dei risultati

### 1) Creazione dell’environment Conda `opt`

Se il file `optimization.yml` contiene già `name: opt`:

```bash
conda env create -f optimization.yaml
conda activate opt
```

In alternativa, per forzare il nome:

```bash
conda env create -n opt -f optimization.yaml
conda activate opt
```

### 2) Lancio degli esperimenti

Esegui:

```bash
python optimization.py
```

> Gli output (accuratezze, tempi, metriche) verranno stampati/ salvati secondo quanto previsto dallo script.

## Rappresentazione dei grafici

Per problemi di rendering grafico sulle macchine dell’Università, i grafici sono stati generati in **Google Colab** utilizzando la funzione `plot_methods_comparison`.
Notebook Colab per la rappresentazione grafica:
👉 [Apri su Google Colab](https://colab.research.google.com/drive/1s8AJa46PFZQc8DH2OVR-neCOXgTnVibg)
