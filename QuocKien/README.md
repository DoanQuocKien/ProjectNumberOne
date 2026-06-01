# Personalized Item Recommendation (PIR) - CS116

This folder contains the complete machine learning pipeline for the Personalized Item Recommendation project by Đoàn Quốc Kiên (24520879).

## Project Overview

The project has evolved significantly from the initial heuristic baseline into a highly optimized, zero-copy streaming architecture using **Polars**, **PyArrow**, and **LightGBM (lambdarank)**.

The system handles a 35GB raw dataset within a strictly constrained Kaggle environment (30GB RAM, 12-hour timeout) by utilizing Reciprocal Rank Fusion (RRF) for candidate generation and a $\mathcal{O}(N)$ memory scaling batch pipeline for feature assembly and inference.

For an exhaustive, 20+ page mathematical deep dive into the 50 hypotheses, data exploration autopsies, and the final pipeline architecture, please read the [Comprehensive Journey Report](file:///d:/CS116/ProjectNumberOne/QuocKien/comprehensive_journey_report.md).

## Reproducing the Submission

The final pipeline and the resulting Kaggle submission files can be completely reproduced by running the finalized Jupyter Notebooks located in the `Refresh` directory:

1. **`Refresh/01_Train_GPU.ipynb`**: This notebook handles the zero-copy streaming of candidates, dynamic assembly of the Polars Lookup Tables (LUTs) with 150+ dense features, and the LightGBM `lambdarank` model training.
2. **`Refresh/02_Inference_CPU.ipynb`**: This notebook loads the trained LightGBM model and streams the 35GB holdout set through a single-pass inference engine, executing block-by-block truncation to output the final Top-12 predictions without breaching RAM limits.

*Note: The candidate generation code and Optuna tuning scripts are also preserved in the `Refresh` directory.*
