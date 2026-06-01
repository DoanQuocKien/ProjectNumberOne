# Comprehensive Recommender System Architecture Report

This document is a highly technical deep-dive into the recommender system pipeline. It completely details the mathematical strategies, feature engineering logic, implementation bottlenecks, and data hypotheses used to build the final ranking engine.

---

## 1. Candidate Generation Strategy (Recall Stage)
**Objective:** Reduce the global search space from roughly 32,000 items down to a dense, high-probability subset of candidates (106 to 230 items per user).

### 1.1 Implemented Retrieval Strategies
The candidate pool for each user is constructed using an ensemble of distinct algorithms, executed via `polars` for memory efficiency:

1. **Session Co-visitation (Collaborative Filtering):**
   - **Mechanism:** Builds a bipartite graph of item-to-item transitions based on users purchasing items within the same session or adjacent timeframes.
   - **Weighting:** The co-occurrence edges are weighted by recency. A co-purchase that happened 2 days ago is weighted higher than one from 6 months ago using a logarithmic decay function.
   - **Extraction:** If a user recently purchased Item A, the system queries the graph and injects the top *K* items most frequently co-purchased with Item A into their candidate pool.

2. **Personal History (Repurchase Dynamics):**
   - **Mechanism:** The user's historical purchases are extracted and re-injected as candidates. 
   - **Weighting:** Items are ranked based on the total frequency of purchase and the recency of the last purchase.

3. **Global & Local Popularity:**
   - **Mechanism:** Acts as a fallback for cold-start users (users with zero history).
   - **Weighting:** Items are ranked by absolute volume over the trailing 4 weeks.

### 1.2 Reciprocal Rank Fusion (RRF)
Because the retrieval strategies output drastically different raw scores (e.g., co-visitation edge weights vs. raw popularity counts), they cannot be natively summed. We utilize **Reciprocal Rank Fusion (RRF)** to normalize and combine them:

$$ RRF\_Score(item) = \sum_{strategy \in S} \frac{1}{60 + rank_{strategy}(item)} $$

Where the constant $k = 60$ prevents the highest-ranked items in any single strategy from dominating the fused score. The candidates are sorted by their final $RRF\_Score$, and only the Top $N$ (e.g., 230) are retained per user.

---

## 2. Feature Engineering (Precision Stage)
**Objective:** Transform the sparse candidates into dense feature matrices for tree-based ranking. The features were heavily inspired by the hypotheses validated in the `data_analysis` folder.

All features are precomputed into **Lookup Tables (LUTs)** using `polars` to avoid redundant calculations during the dataset join phase.

### 2.1 Customer Clustering & Diversity (Based on Ideas 8, 41, 46)
* **Customer Lifecycle Stage:** Categorizing users based on tenure into New (<30 days), Active, or Dormant (>90 days without purchase). (Idea 41)
* **Diversity Entropy:** Quantifying whether a user is an "Explorer" or a "Habitual Shopper" by calculating the Shannon Entropy of their historical category distribution:
  $$ Entropy = -\sum (p_{category} \times \log(p_{category})) $$
  A high entropy user (Explorer) triggers the model to favor cross-category candidates. (Idea 8, 46)
* **Archetype Projection:** We construct a sparse `User × Category` matrix of interaction counts. We apply `TruncatedSVD` (components = 16) to compress this into dense latent vectors. The **Cosine Similarity** between the User Archetype vector and the Candidate Item's category vector is used as a direct affinity feature.

### 2.2 Category & Brand Affinity (Based on Ideas 2, 42, 43)
* **Anchor Categories:** We found that 82% of users have an "Anchor Category" (Herfindahl-Hirschman Index > 0.3). Features like `user_category_lift` measure how much a user prefers the candidate's category relative to the global average. (Idea 42)
* **Brand Loyalty Matrices:** Calculating the HHI of brand purchases per user. Loyalists are heavily constrained to their preferred brands, especially within their anchor categories. A boolean feature `is_historical_brand` acts as a massive boost for loyalists. (Idea 2, 43)

### 2.3 Pricing & Economic Sensitivity (Based on Ideas 10, 14, 35)
* **Price Tier Movement:** Users were segmented into Budget, Standard, and Premium tiers. 
* **Relative Price Ratios:** We calculate the user's historical median purchase price ($P_{median}$). The feature `candidate_price_ratio` is calculated as $\frac{P_{candidate}}{P_{median}}$. If this ratio is > 3.0, the tree penalizes the item, preventing a budget shopper from being recommended a premium luxury stroller. (Idea 14)
* **Location Price Delta:** Adjusting recommendations based on whether the specific store location skews towards budget or premium items. (Idea 10)

### 2.4 Replenishment & Repeat Propensity (Based on Ideas 6, 17, 44)
* **Category-Specific Cycles:** Consumables like Milk and Diapers have massive repeat propensities (up to 32% with a 13-day median gap). 
* **Replenishment Feature:** We calculate the exact `days_since_last_purchase`. If this aligns with the category's `expected_replenishment_gap`, the model learns to push the item to Rank 1. (Idea 6, 44)
* **One-Off Penalties:** For discretionary categories (Fashion, Toys) where repeat propensity is near zero, a boolean flag `previously_purchased_discretionary` allows the model to actively penalize its future ranking. (Idea 17)

### 2.5 Size Ladders & Child Aging (Based on Idea 22, 34)
* **Size Progression Vectors:** Diapers and Apparel follow a strict chronological size ladder. By calculating the user's `tenure_days`, the model learns the implicit correlation between customer age and the required size tier (e.g., shifting from Size M to Size L). (Idea 34)

### 2.6 Location Specificity & Assortment (Based on Ideas 36, 37, 38)
* **Location Availability Gaps:** Average locations only stock 10% of the global catalog. 
* **Availability Feature:** We map the user's `primary_location`. If the candidate item has exactly 0 historical sales at that location, the `location_item_sales` feature is 0, allowing the model to suppress "Ghost Items" that the user cannot physically buy. (Idea 36, 38)

---

## 3. LightGBM Ranker Engine & Engineering Architecture

The machine learning framework utilizes Microsoft's LightGBM in `lambdarank` mode to order the generated candidates.

### 3.1 Model Objective & Tuning
* **Objective Function:** `lambdarank` optimized for `ndcg@10`. The model learns not just binary classification, but the optimal *order* of items to maximize hits in the top 10 slots.
* **Hyperparameter Tuning:** An integrated `Optuna` loop utilizes `lgb.cv` (3-fold cross-validation) to find the mathematically optimal tree structures.
  * `num_leaves`: Bounded between 31 and 255.
  * `learning_rate`: Bounded between 0.01 and 0.2.
  * `min_data_in_leaf`: Bounded between 20 and 500 to prevent overfitting on niche items.

### 3.2 Overcoming the Training RAM Explosion (`np.vstack`)
A critical implementation flaw in standard dataset building is dynamic array concatenation. Iterating through user chunks and calling `np.vstack()` repeatedly forces Python to clone multi-gigabyte matrices in RAM, causing instant OOMs.
* **The Solution:** The dataset builder utilizes `pyarrow.parquet.ParquetFile.iter_batches()`. It parses the flat candidate schema sequentially, computes features into lightweight `numpy.float32` arrays, and appends them to a native Python `list`. A single `np.vstack()` is executed exactly once at the end of the loop, reducing memory complexity from $O(N^2)$ array copying down to $O(N)$.

### 3.3 Vectorized Inference Engine (The I/O Bottleneck)
Scoring 2.8 million users for the final submission initially hit a catastrophic 4.6-hour bottleneck.
* **The Problem:** Standard dataframe loading (`pl.scan_parquet(file).filter(users).collect()`) forced the hard drive to scan the entire 35GB disk file for *every single user chunk*. At 560 chunks, the disk was being scanned 560 times.
* **The PyArrow Streaming Solution:** The inference loop was rewritten to stream the disk sequentially from top to bottom exactly **one time**.
  * `iter_batches(batch_size=2_000_000)` pulls exactly 2 million candidate rows into memory.
  * Features are calculated instantly via Polars.
  * `model.predict()` scores the batch on CPU.
  * The dataframe is grouped by `customer_id`, sorted by `score`, and only the **Top 10** items are retained.
  * `del df, X; gc.collect()` completely shreds the massive feature matrices, returning RAM to near zero before the next batch is pulled.
* **Result:** Inference time dropped from 4.6 hours down to approximately 15 minutes.

---

## 4. Unimplemented Features (Out of Time)

While the pipeline captures the strongest signals, several promising ideas from the analysis phase were left out due to strict time and compute constraints:

1. **Competitive Item Substitution (Idea 29):** 
   - *Concept:* We identified 1,400+ highly competitive pairs (e.g., Enfa vs. Similac) through negative correlation. The plan was to apply a post-processing diversity filter to avoid recommending redundant substitute items in the same Top 10 slots.
   - *Reason:* Building the substitution matrix and applying iterative post-processing diversity logic was too computationally heavy for the inference time limit.
2. **Hidden Mission Clusters (Idea 27):** 
   - *Concept:* Grouping disparate items into inferred "Shopping Missions" (e.g., Maternity Prep, Weekend Outing) based on co-occurrence, and boosting entire clusters if one item was engaged.
   - *Reason:* Required running complex unsupervised clustering (K-Means/DBSCAN) on sparse item vectors, which was skipped to focus on direct collaborative filtering.
3. **Category Trend & Lifecycle Momentum (Ideas 9, 31, 39):**
   - *Concept:* Calculating the rolling 30-day and 90-day trajectory slopes of brands and categories to boost "Market Disruptors" and penalize "Declining Legacy Brands."
   - *Reason:* Required complex time-series aggregations that slowed down the feature engineering step significantly.
4. **Graph Embeddings / Node2Vec:** 
   - *Concept:* We considered training a Graph Neural Network (or Word2Vec equivalent) on the user-item bipartite graph to extract latent vectors and use approximate nearest neighbors (Faiss) for candidate retrieval. 
   - *Reason:* The immense RAM overhead required to train embeddings on 2.8 million users exceeded the 30GB Kaggle hardware limits.
