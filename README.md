# CS116: Python for Machine Learning - Personalized Item Recommendation (PIR)

This repository contains the final submissions for the Personalized Item Recommendation project.

## Final Results & Submissions

The primary, highest-scoring solution is located in the **`QuocKien`** directory.

The team achieved the following **Precision@10** scores on the Kaggle leaderboard:

*   **QuocKien:** `0.0924` (Winning Solution)
*   **Hung:** `0.0881`
*   **Phong:** `0.0874`

### How to Reproduce the Winning Solution

To reproduce the `0.0924` score, please navigate to the `QuocKien` directory. 
The complete zero-copy streaming architecture using Polars and LightGBM (`lambdarank`) is documented there. 

You can run the finalized Kaggle pipeline using the two notebooks in the `Refresh` folder:
1. `QuocKien/Refresh/01_Train_GPU.ipynb`
2. `QuocKien/Refresh/02_Inference_CPU.ipynb`

For a mathematical breakdown of the models, feature engineering, and failed hypotheses, please read the [Comprehensive Journey Report](QuocKien/comprehensive_journey_report.md).
