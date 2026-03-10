# Choice Models and Permutation Invariance

This repository contains the official code for the paper  
**"Choice Models and Permutation Invariance: Demand Estimation in Differentiated Products Markets"**  
[SSRN Working Paper #4508227](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4508227)

---

## Overview

This repository contains two parts:
1) A tutorial on how to use the method proposed in the paper: Inference.ipynb — This notebook provides a tutorial that replicates the main experiment from Section 3.4 of the paper. It covers both the plug-in and debiased estimation methods. Users can replace the generated data with real data to apply both the plug-in and debiased methods proposed in the paper.

All necessary functions are defined in the following supporting scripts:

| File / Folder | Description |
|---------------|-------------|
| `prediction.py` | Contains standard data-generation routines, the permutation-invariant neural network, training logic, and plug-in evaluation helpers. |
| `debiase.py`    | Implements the debiasing procedure: moment construction, training for debiased term, and final inference. |
|  `results/` | A directory to store intermediate results from 100 random simulation draws of inference. |


2) Replication files for all the tables and figures in the paper and the web appendix. `Replication/`

In this folder, each notebook corresponds to a specific table or figure, as indicated in the notebook names.

/src contains the source files for the necessary functions used in the numerical experiments and empirical analysis, we detailed below. 

| File / Folder | Description |
|---------------|-------------|
| `data_generation.py` | Contains various data-generation processes, including RCL with inattention, RCL with non-linear utility, RCL and MNL with fixed effects, random coefficients that follow a bimodal mixture distribution, and prices with reduced variance, etc.|
| `neural_network.py`    | Contains the neural network architectures of our proposed method and the non-parametric method used as a comparison in the paper.|
| `estimation.py`    |Includes the training and evaluation procedures for all methods regarding market share and elasticity estimation in the paper.|
| `train_varying_products.py`    |  Includes the training and evaluation procedures of the proposed method that accommodate the issue of varying products in each market. This code is used to run the empirical analysis on the U.S. automobile dataset.|

Result_Tables/ is used to store intermediate results for the numerical experiments. We provide these files to reproduce the tables presented in the paper, as running each numerical experiment may take a long time. Users can also generate similar tables by running the simulations themselves. 

Note: Running the simulations on different hardware or software environments may lead to slightly different numerical results due to randomness in training and system-level differences. However, these minor variations should not affect the qualitative findings or the main conclusions of the paper.


In addition,
  - `requirements.txt` lists all required Python packages. 
  
---

## Quick start

```bash
git clone https://github.com/yeliu929/choice_model_pi.git
cd choice_model_pi
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

jupyter lab inference.ipynb   # Launch the tutorial
```

---

If you find this repository useful for your research, please consider citing the following paper:

```bibtex
@article{singh2023choice,
  title     = {Choice Models and Permutation Invariance: Demand Estimation in Differentiated Products Markets},
  author    = {Singh, Amandeep and Liu, Ye and Yoganarasimhan, Hema},
  journal   = {arXiv preprint arXiv:2307.07090},
  year      = {2025}
}
```
---
Contact: yeliu@uw.edu
