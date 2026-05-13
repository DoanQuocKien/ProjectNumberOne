# Analytical Insights Master List (Recommendation Engine Requirements)

This document serves as the primary source of truth for translating data exploration findings into production model requirements, strictly aligned with the PIR (Time-aware, Cold-start) evaluation metrics.

## Summary Table

| ID | Title | Decision | PIR Score | Primary Driver |
|----|-------|----------|-----------|----------------|
| 1 | Holiday Effect | DROP | 0.00 | p-values > 0.05 |
| 2 | Brand/Manufacturer Commitment | KEEP | 0.90 | High HHI for loyalists |
| 3 | Category Connections | PARTIAL KEEP | 0.60 | category_l1 lift > 2.7 |
| 4 | Size Normalization | PARTIAL KEEP | 0.50 | Replenishment for 1.3% of catalog |
| 5 | Purchase Seasonality | KEEP | 0.75 | Significant weekday (Sunday) peaks |
| 6 | Replenishment Cycle | KEEP | 1.00 | 22-day median gap |
| 7 | Basket Structure (Rules) | PARTIAL KEEP | 0.60 | category_l1 bundles |
| 8 | Customer Diversity Segment | KEEP | 0.85 | 3 distinct entropy segments |
| 9 | Item Lifecycle Stage | KEEP | 0.80 | Growth (+368 slope) vs Decline |
| 10 | Price Sensitivity | KEEP | 0.85 | Location-level price deviations |
| 11 | Item Popularity by Segment | CONDITIONAL KEEP | 0.70 | Weak segment item isolation |
| 12 | Category Stability | DROP | 0.00 | Spearman rank -0.44 |
| 13 | Discovery Path | KEEP | 0.80 | 22.4x lift for in-cat discovery |
| 14 | Item Affinity by Price Tier | KEEP | 0.75 | Variance > 0.2 in 69.5% combos |
| 15 | Brand Switching | PARTIAL KEEP | 0.60 | High entropy destinations |
| 16 | Location Preference | DROP | 0.00 | HHI = 0.0016 |
| 17 | Repeat Propensity | KEEP | 0.95 | Bimodal: Staples vs One-Offs |
| 18 | Cross-Category Affinity | KEEP | 0.85 | Thematic clusters |
| 19 | Item Lifecycle H2 vs H1 | KEEP | 0.85 | New items = Acquisition |
| 20 | Manufacturer Affinity by Tier | KEEP | 0.85 | HHI 0.61 within combos |
| 21 | Location-Category Interaction | KEEP | 0.75 | Mall vs Convenience stores |
| 22 | Size Tier Affinity | KEEP | 0.90 | 61.9% diff in avg age size |
| 23 | Item Momentum | KEEP | 0.95 | 1.58x lift for High_Low |
| 24 | Customer-Location Specificity | KEEP | 0.80 | 2.8% Jaccard overlap |
| 25 | Brand-Price Interaction | KEEP | 0.80 | 64% variance for Known Brands |
| 26 | Tenure-Lifecycle | KEEP | 0.90 | 2.06x lift for New items |
| 27 | Hidden Clusters | KEEP | 0.85 | 3.34x cohesion lift |
| 28 | Exploration Score | KEEP | 0.90 | Exploration Rate variance |
| 29 | Competitive Item Substitution | KEEP | 0.85 | 1,401 competitive pairs found (Strength up to 34.1) |
| 30 | Location Ranking Variance | KEEP | 0.85 | High variance found (Avg Tau 0.61). |
| 31 | Category Trend Curves | KEEP | 0.75 | 7.1% Rising, 11.9% Declining categories. |
| 32 | Brand Trend Within Category | KEEP | 0.80 | Disruptors gaining up to 12% share/mo. |

---

## Detailed Analytical Requirements

### Idea 2: Brand/Manufacturer Commitment
- **Finding:** High-frequency customers have high HHI for specific brands, indicating intense commitment.
- **Usefulness Score:** **0.90 / 1.0** (Based on PIR time-aware & cold-start requirements)
- **Actionable Insight (Feature Engineering):** Calculate a `brand_loyalty_score` (HHI of brands per customer).
- **Actionable Insight (Ranking Logic):** If a customer's `brand_loyalty_score` > 0.7 for a category, use a **Hard Filter** to show only that brand's items in the top 3 slots.
- **Future Exploration (Sub-Idea):** Test if brand loyalty decays over time or shifts abruptly after a bad experience/churn event.

### Idea 3: Category Connections
- **Finding:** Item-level and L3-level co-purchases have extreme lift but tiny support (1-4 occurrences). Only `category_l1` shows stable, usable associations (e.g., Textile <-> Fashion lift ~2.96).
- **Usefulness Score:** **0.60 / 1.0** (Based on PIR time-aware & cold-start requirements)
- **Actionable Insight (Feature Engineering):** Create a `l1_co_purchase_matrix`.
- **Actionable Insight (Ranking Logic):** Use category_l1 connections for 'Gateway' cross-selling. Do not use exact-item associations from this method due to sparsity.
- **Future Exploration (Sub-Idea):** Explore sequential connection timing (e.g., how many days after Category A does Category B get bought?).

### Idea 4: Size Normalization
- **Finding:** Normalization coverage is extremely low (1.3%) due to 'Không xác định' labels. Within that 1.3%, customers rebuy the *same size* 90.7% of the time, rather than sizing up.
- **Usefulness Score:** **0.50 / 1.0** (Based on PIR time-aware & cold-start requirements)
- **Actionable Insight (Feature Engineering):** Create a `days_since_last_size_buy` counter.
- **Actionable Insight (Ranking Logic):** Apply replenishment reminders for diapers and clothes using the exact same size the user bought last time. Broad 'size up' logic is not supported by the data.
- **Future Exploration (Sub-Idea):** Investigate if 'Unknown' size labels can be inferred via NLP on the item name text.

### Idea 5: Purchase Seasonality
- **Finding:** Weekday seasonality is highly significant (p~0.0). Sunday is the peak (1.22x txn lift) and Thursday is the trough. November is the peak month.
- **Usefulness Score:** **0.75 / 1.0** (Based on PIR time-aware & cold-start requirements)
- **Actionable Insight (Feature Engineering):** Extract `weekday` and `month` context features.
- **Actionable Insight (Ranking Logic):** Increase the 'Exploration' temperature slightly on weekends when traffic and basket sizes are higher. Boost Q4 trending items starting in October.
- **Future Exploration (Sub-Idea):** Cross-reference weekday peaks with specific locations (e.g., do mall stores peak on Sundays while street stores peak on Mondays?).

### Idea 6: Replenishment Cycle
- **Finding:** Extremely strong signal: 17.1% of pairs are repeats. Median gap is 22 days. Formula is ultra-predictable (12-15 day cycles).
- **Usefulness Score:** **1.00 / 1.0** (Based on PIR time-aware & cold-start requirements)
- **Actionable Insight (Feature Engineering):** Compute `expected_replenishment_date` per user-category.
- **Actionable Insight (Ranking Logic):** When `current_date` > `expected_replenishment_date`, force the specific item to Rank #1. This perfectly solves the time-aware metric.
- **Future Exploration (Sub-Idea):** Calculate standard deviation of replenishment gaps per individual customer rather than relying solely on global category medians.

### Idea 7: Basket Structure (Rules)
- **Finding:** Similar to Idea 3, item-level rules have 5,000x lift but sparse support. Category_l1 (Accessories <-> Fashion) is stable.
- **Usefulness Score:** **0.60 / 1.0** (Based on PIR time-aware & cold-start requirements)
- **Actionable Insight (Feature Engineering):** N/A
- **Actionable Insight (Ranking Logic):** Only use for fallback generic bundles at the L1 level.
- **Future Exploration (Sub-Idea):** Test negative association rules (items that are explicitly *never* bought together).

### Idea 8: Customer Diversity Segment
- **Finding:** Customers neatly split into Narrow-Repeat (avg 1.2 distinct items), Balanced, and Broad-Explorer (avg 11.0 items).
- **Usefulness Score:** **0.85 / 1.0** (Based on PIR time-aware & cold-start requirements)
- **Actionable Insight (Feature Engineering):** `customer_diversity_segment` (Narrow/Balanced/Broad).
- **Actionable Insight (Ranking Logic):** For 'Narrow' users, optimize heavily for repetition. For 'Broad' users, penalize showing items from the same category to force discovery.
- **Future Exploration (Sub-Idea):** Track whether a customer's diversity segment drifts over their lifecycle (e.g., do they start as explorers and become narrow repeaters?).

### Idea 9: Item Lifecycle Stage
- **Finding:** Growth items (+368 slope) and Decline items (-281 slope) are completely separable and constitute ~45% of the catalog.
- **Usefulness Score:** **0.80 / 1.0** (Based on PIR time-aware & cold-start requirements)
- **Actionable Insight (Feature Engineering):** Calculate 3-month moving average slope (`item_momentum`).
- **Actionable Insight (Ranking Logic):** Filter out 'Decline' items from the top 20 slots unless the user has explicitly bought them before. Boost 'Growth' items for Cold Start users.
- **Future Exploration (Sub-Idea):** Analyze the geographic spread of new items (do they launch in HCM first before spreading?).

### Idea 10: Price Sensitivity
- **Finding:** The tier mix is strongly skewed to budget. Location price deviations exist (median 3.1%). Strongly deviated prices alter purchase behavior.
- **Usefulness Score:** **0.85 / 1.0** (Based on PIR time-aware & cold-start requirements)
- **Actionable Insight (Feature Engineering):** `location_price_delta_percentile` for the current store.
- **Actionable Insight (Ranking Logic):** If the current location has a positive price deviation (expensive), artificially boost 'Budget' tier items for that specific session.
- **Future Exploration (Sub-Idea):** Correlate price sensitivity with the specific day of the month (e.g., payday effects).

### Idea 11: Item Popularity by Segment
- **Finding:** Every segment's top items have a global lift < 1.0, meaning bestsellers are global, not deeply segment-specialized.
- **Usefulness Score:** **0.70 / 1.0** (Based on PIR time-aware & cold-start requirements)
- **Actionable Insight (Feature Engineering):** `global_popularity_rank`.
- **Actionable Insight (Ranking Logic):** Do not attempt naive segment-specific bestsellers. Rely entirely on Global Popularity to solve the Jan 2026 Cold Start problem.
- **Future Exploration (Sub-Idea):** Re-run popularity ranking within specific geographic regions instead of global.

### Idea 13: Discovery Path
- **Finding:** Customers discover new items in their existing top 3 categories 22.4x more often than random.
- **Usefulness Score:** **0.80 / 1.0** (Based on PIR time-aware & cold-start requirements)
- **Actionable Insight (Feature Engineering):** Identify user's `top_3_categories`.
- **Actionable Insight (Ranking Logic):** When recommending 'New' items, constrain them to the user's existing Top 3 categories rather than throwing random cross-category ideas.
- **Future Exploration (Sub-Idea):** Analyze the 'churn path' (which category is typically the last one bought before a user goes dormant?).

### Idea 14: Item Affinity by Price Tier
- **Finding:** 69.5% of combinations show variance > 0.20. Low_Low segments allocate heavily to budget, while High_High allocates 22% to premium.
- **Usefulness Score:** **0.75 / 1.0** (Based on PIR time-aware & cold-start requirements)
- **Actionable Insight (Feature Engineering):** `user_premium_spend_ratio`.
- **Actionable Insight (Ranking Logic):** Hard-constraint on price tier for low-frequency shoppers (Low_Low) to ensure conversion via low price-points.
- **Future Exploration (Sub-Idea):** Test if price tier upgrades happen sequentially as a child ages.

### Idea 15: Brand Switching
- **Finding:** Switching destinations are fragmented (entropy 2.05). However, baseline brand loyalty is huge (e.g., 53% for Meiji).
- **Usefulness Score:** **0.60 / 1.0** (Based on PIR time-aware & cold-start requirements)
- **Actionable Insight (Feature Engineering):** `last_brand_bought_in_cat`.
- **Actionable Insight (Ranking Logic):** Use 'Last Brand Bought' to boost repeats, but do not use Markov-chain switching paths as they are too noisy.
- **Future Exploration (Sub-Idea):** Map brand switching paths specifically after stock-out events to identify true substitutes.

### Idea 17: Repeat Propensity
- **Finding:** Global item repeat is 3.3%, but staples (Formula/Diapers) hit 32-45%. This is a massive deterministic signal.
- **Usefulness Score:** **0.95 / 1.0** (Based on PIR time-aware & cold-start requirements)
- **Actionable Insight (Feature Engineering):** `item_repeat_propensity_score`.
- **Actionable Insight (Ranking Logic):** Bifurcate the algorithm: If an item's repeat propensity is <0.1, never recommend it again once bought. If >0.3, anchor it at the top.
- **Future Exploration (Sub-Idea):** Measure if repeat propensity increases when the item is purchased on a promotion.

### Idea 18: Cross-Category Affinity
- **Finding:** 542 pairs show lift > 1.5, forming 'need state' clusters (e.g., Maternity Pads + Mom Items).
- **Usefulness Score:** **0.85 / 1.0** (Based on PIR time-aware & cold-start requirements)
- **Actionable Insight (Feature Engineering):** Create `need_state_cluster_id` mapping.
- **Actionable Insight (Ranking Logic):** If an item from a need-state is added to cart, immediately upweight all other categories in that specific cluster.
- **Future Exploration (Sub-Idea):** Build a graph neural network representation of these clusters for vector embedding.

### Idea 19: Item Lifecycle H2 vs H1
- **Finding:** High_Low (recent acquisitions) over-index on 'New' items with a 3.27x lift.
- **Usefulness Score:** **0.85 / 1.0** (Based on PIR time-aware & cold-start requirements)
- **Actionable Insight (Feature Engineering):** `item_launch_age_days`.
- **Actionable Insight (Ranking Logic):** For users with tenure < 30 days, heavily skew the top 5 recommendation slots toward newly launched items.
- **Future Exploration (Sub-Idea):** Analyze how long an item typically stays in the 'Growth' phase before stabilizing.

### Idea 20: Manufacturer Affinity by Tier
- **Finding:** Once a segment and price tier are chosen, manufacturer concentration is extreme (HHI 0.61).
- **Usefulness Score:** **0.85 / 1.0** (Based on PIR time-aware & cold-start requirements)
- **Actionable Insight (Feature Engineering):** `dominant_manufacturer_for_segment_tier`.
- **Actionable Insight (Ranking Logic):** If we know a user's segment and preferred tier, hard-boost the dominant manufacturer for that combo.
- **Future Exploration (Sub-Idea):** Test cross-tier manufacturer loyalty (does buying premium Abbott predict buying budget Abbott?).

### Idea 21: Location-Category Interaction
- **Finding:** Mall locations have 6.9x lift for Apparel. Convenience stores over-index on Beverages.
- **Usefulness Score:** **0.75 / 1.0** (Based on PIR time-aware & cold-start requirements)
- **Actionable Insight (Feature Engineering):** `location_store_type` (Mall/Convenience).
- **Actionable Insight (Ranking Logic):** If the active session is at a Mall store, boost 'Showroom' categories (Apparel, Gear). If Convenience, boost Consumables.
- **Future Exploration (Sub-Idea):** Cluster locations into 'Archetypes' purely based on their category sales vectors.

### Idea 22: Size Tier Affinity
- **Finding:** High_High segments buy toddler sizes (1.7y avg), while Low_Low churners buy infant sizes (1.0y avg).
- **Usefulness Score:** **0.90 / 1.0** (Based on PIR time-aware & cold-start requirements)
- **Actionable Insight (Feature Engineering):** `predicted_child_age_from_segment`.
- **Actionable Insight (Ranking Logic):** Use RFM segment as a proxy for child age. Filter out Toddler sizes for Low_Low users entirely.
- **Future Exploration (Sub-Idea):** Create a predictive 'Next Size' model based on historical transitions for the specific child.

### Idea 23: Item Momentum
- **Finding:** Trending items have massive 1.58x lift for new acquisitions, but 0.75x lift for churning loyalists.
- **Usefulness Score:** **0.95 / 1.0** (Based on PIR time-aware & cold-start requirements)
- **Actionable Insight (Feature Engineering):** `30d_momentum`.
- **Actionable Insight (Ranking Logic):** Apply Momentum Multiplier exclusively to New/Infrequent users. Disable momentum logic for core loyalists.
- **Future Exploration (Sub-Idea):** Test if momentum is driven by viral external events (TikTok, holidays) or purely internal discovery.

### Idea 24: Customer-Location Specificity
- **Finding:** Customers buy mutually exclusive baskets at different locations (Jaccard = 2.8%).
- **Usefulness Score:** **0.80 / 1.0** (Based on PIR time-aware & cold-start requirements)
- **Actionable Insight (Feature Engineering):** Build user profiles by `(customer_id, location)`.
- **Actionable Insight (Ranking Logic):** When ranking for a specific location visit, history from *other* locations must be down-weighted by >90%.
- **Future Exploration (Sub-Idea):** Analyze if users shopping at multiple locations are higher LTV than single-location users.

### Idea 25: Brand-Price Interaction
- **Finding:** Known brands have extreme segment loyalty. 'Không xác định' (Unknown) items are universally treated as budget fillers.
- **Usefulness Score:** **0.80 / 1.0** (Based on PIR time-aware & cold-start requirements)
- **Actionable Insight (Feature Engineering):** Flag `is_known_brand`.
- **Actionable Insight (Ranking Logic):** Do not attempt segment-based personalization for Unbranded items. Only surface them when sorting by lowest price.
- **Future Exploration (Sub-Idea):** Test if unbranded items have higher elasticity during major sales events compared to branded items.

### Idea 26: Tenure-Lifecycle
- **Finding:** New customers (<90d) have 2.06x lift for 'New' items. Established customers have 0.09x lift (effectively zero).
- **Usefulness Score:** **0.90 / 1.0** (Based on PIR time-aware & cold-start requirements)
- **Actionable Insight (Feature Engineering):** `user_tenure_days`.
- **Actionable Insight (Ranking Logic):** Hard cut-off at 90 days: Stop showing 'New Arrivals' in top slots and pivot entirely to 'Stable' items.
- **Future Exploration (Sub-Idea):** Track the exact week where the novelty curve drops off (is it exactly 90 days, or a gradual decline?).

### Idea 27: Hidden Clusters
- **Finding:** Discovered 5 hidden mission clusters (e.g., Feeding, Hygiene) with massive 3.34x cohesion.
- **Usefulness Score:** **0.85 / 1.0** (Based on PIR time-aware & cold-start requirements)
- **Actionable Insight (Feature Engineering):** Map `item_id` to `mission_cluster_id`.
- **Actionable Insight (Ranking Logic):** Cross-Category Mission Boost: Adding an item triggers a boost for all other L2 categories in that same hidden cluster.
- **Future Exploration (Sub-Idea):** Perform a basket-level association rule mining exclusively within a single discovered mission cluster.

### Idea 28: Exploration Score
- **Finding:** Milk is purely habitual (30% explore rate). Fashion/Gear is purely exploratory (>95% explore rate).
- **Usefulness Score:** **0.90 / 1.0** (Based on PIR time-aware & cold-start requirements)
- **Actionable Insight (Feature Engineering):** `category_novelty_coefficient`.
- **Actionable Insight (Ranking Logic):** Dynamic Temperature: For Milk, heavily penalize unpurchased items. For Fashion, heavily penalize previously purchased items.
- **Future Exploration (Sub-Idea):** Correlate exploration score with the total lifetime value of the customer.

### Idea 29: Competitive Item Substitution
**Usefulness Score:** 0.85
**Status:** KEEP

#### 1. Empirical Findings
- Found **1,401 strong competitive substitution pairs** (Strength > 1.5) across 123 categories.
- **Strongest Signals:** Found in 'Enfa' and 'Nestle' milk categories.
  - *Example:* `Enfa A+` vs `Enfa Enspire` has a substitution strength of **34.1**, meaning they are bought together 35x less than random chance would suggest.
  - *Abbott:* `Abbott Grow` vs `Similac` also shows significant substitution (Strength 14.2).

#### 2. Feature Engineering Requirements
- **Substitution Lookup Table:** Build a mapping of `(item_id -> list of substitutes)` where substitution strength > 2.0.

#### 3. Ranking Logic
- **Diversity Filter:** If the #1 ranked item for a customer is Item A, apply a **0.2x penalty** to all items in its substitution list within the Top 10 recommendations to ensure the slot is not wasted on a redundant alternative.

#### 4. Future Exploration (Sub-Idea)
- **Time-Decayed Substitution:** Does a customer switch substitutes over time (e.g., moving from Brand A to Brand B as they age out of a specific product tier)?

### Idea 30: Multi-Location Ranking Variance
**Usefulness Score:** 0.85
**Status:** KEEP

#### 1. Empirical Findings
- **Low Correlation:** Kendall's Tau correlation between Global and Local ranks is significantly low, ranging from **0.46** (HCM Nguyễn Trãi) to **0.71** (BDU Pasteur).
- **Massive Rank Shifts:** Found **"Local Heroes"** with rank deltas exceeding **700 positions**.
  - *Example:* Item `1920001360751` is rank 885 globally but jumps to **rank 124** in the Binh Duong (BDU) location.
- **Average Rank Delta:** Top items see an average absolute shift of **150–300 positions** across the 10 largest locations.

#### 2. Feature Engineering Requirements
- **Local Popularity Score:** Compute `local_vs_global_lift` for the top 500 items per location.
- **Store Cluster:** Group locations with similar Tau-profiles into store clusters for cold-start location ranking.

#### 3. Ranking Logic
- **Regional Boost:** If a customer's `last_known_location` is available, apply a **1.25x boost** to items with a `local_rank < 50` in that location to surface regional favorites.

#### 4. Future Exploration (Sub-Idea)
- **Assortment-Adjusted Ranking:** Does the variance disappear if we only rank items that are *actually in stock* at that location (i.e., is it a preference difference or an availability difference)?

### Idea 31: Category Trend Curves
**Usefulness Score:** 0.75
**Status:** KEEP

#### 1. Empirical Findings
- **Rising Categories:** Found several categories with strong growth slopes (normalized > 0.1).
  - *Sữa chua uống:* Highest momentum (+0.25 slope).
  - *Snow Brand:* +0.20 slope (R² 0.58).
  - *TPCN cho mẹ:* +0.11 slope.
- **Declining Categories:** Significant decline in legacy brands and household hygiene.
  - *Goon:* -0.54 slope.
  - *Pampers:* -0.30 slope.
  - *Household Hygiene:* -0.26 slope.
- **Trend Distribution:** Approx 19% of the catalog shows non-stable trajectories (either rising or declining).

#### 2. Feature Engineering Requirements
- **Category Momentum Score:** Monthly growth slope (normalized) for the last 6 months.

#### 3. Ranking Logic
- **Momentum Boost:** Apply a **1.15x boost** to items in 'Rising' categories.
- **Sunset Penalty:** Apply a **0.8x penalty** to items in 'Declining' categories to prioritize fresher assortment.

#### 4. Future Exploration (Sub-Idea)
- **Cohort-Specific Trends:** Do new customers drive the rising categories, or is it a shift in behavior from old customers?

### Idea 32: Brand Trend Within Category
**Usefulness Score:** 0.80
**Status:** KEEP

#### 1. Empirical Findings
- **Market Disruptors:** Brands gaining significant share within their categories.
  - *Animo (Car Seats):* Massive +0.12 monthly share gain.
  - *Enterogermina (Supplements):* +0.09 share gain.
  - *Metacare (Nutricare):* +0.06 share gain.
- **Losing Share:** Established brands seeing steady decline.
  - *ColosCare:* -0.09 share slope.
  - *Bledina:* -0.05 share slope.
- **Concentration:** Found that in fragmented categories, disruptors can flip market leadership in under 12 months.

#### 2. Feature Engineering Requirements
- **Brand Share Momentum:** 3-month rolling slope of category market share.

#### 3. Ranking Logic
- **Disruptor Boost:** Apply a **1.2x boost** to items from brands with `share_slope > 0.05`.
- **Legacy Penalty:** Apply a **0.9x penalty** to brands with `share_slope < -0.05` to avoid over-recommending fading favorites.

#### 4. Future Exploration (Sub-Idea)
- **New Launch Impact:** How much of the share gain is driven by new item launches vs. price promotion of existing items?
