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
| 33 | Manufacturer Stability | DROP | 0.00 | Mfg is 22.1% more volatile than Brands. |
| 34 | Size Ladder by Age-Proxy | KEEP | 0.90 | 82% upgrade rate in Diapers (0.6Y->1.2Y). |
| 35 | Price Tier Specialization | KEEP | 0.85 | Named brands have 0.0 median entropy (100% specialized). |
| 36 | Location Assortment Coverage | KEEP | 0.90 | Avg item coverage is only 10.0% per location. |
| 37 | Location-Specific Category Mix | KEEP | 0.85 | Extremely stable (0.948 corr) category-location hubs. |
| 38 | Location-Item Availability Gap | KEEP | 0.90 | Median gap is 96.3%. Fashion items missing from 91% of hubs. |
| 39 | Category-Lifecycle Interaction | KEEP | 0.80 | High churn in Fashion (44% decline) vs Stable Textile (70%). |
| 40 | Item Concentration vs Breadth | KEEP | 0.75 | 3 archetypes: Global Staples, Niche Heroes, Local Staples. |
| 41 | Customer Lifecycle Stage | KEEP | 0.85 | New users discovery (3.52 ent) vs Active staples (3.16 ent). |
| 42 | Category Affinity by Customer | KEEP | 0.90 | 82.2% users have HHI > 0.3. Narrow shoppers (1.66M) vs Broad (199k). |
| 43 | Brand Loyalty Within Categories | KEEP | 0.85 | HHI 0.59 (Inside) vs 0.46 (Outside). Diff 0.127 > 0.1 threshold. |
| 44 | Customer-Item Repeat Patterns | KEEP | 0.95 | Milk (32% rep, 13d gap) vs Fashion (0d gap). Replenishment ready. |
| 45 | Size Progression Behavior | KEEP | 0.80 | Accessories (43% upgrade) vs Diapers (87% stable). Size-ladder found. |
| 46 | Customer Segment Clustering | KEEP | 0.90 | 4 archetypes: Whales (173 units/cust), Habituals (0.97 HHI). |
| 47 | Location Affinity by Customer | KEEP | 0.85 | 85.1% have HHI > 0.5. Median HHI is 1.0. High stationarity. |
| 48 | Price Tier Movement | KEEP | 0.80 | 71% show drift: 35.5% Up, 35.5% Down. High economic mobility. |
| 49 | Purchase Velocity & Seasonality | KEEP | 0.85 | Median 2 visits/mo. Top 25% shop 4+ times. Nov peak. |
| 50 | Cross-Category Segment Patterns | KEEP | 0.90 | Jaccard 0.538. Explorers show 2.2x lift for Membership bundles. |

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

### Idea 33: Manufacturer Stability (Named vs. Unknown Split)
**Usefulness Score:** 0.00
**Status:** DROP

#### 1. Empirical Findings
- **Hypothesis Refuted:** Even after isolating **Named Entities** (removing 'Không xác định' noise), Manufacturers are **22.1% more volatile** (CV 0.395) than Brands (CV 0.324).
- **Lower Predictability:** Aggregating to the manufacturer level actually *decreases* signal stability. This suggests that brand-specific dynamics are the primary drivers of category demand, and manufacturers do not act as a cohesive 'loyalty anchor'.
- **Structural Unknowns:** The 'Không xác định' bucket acts as a background constant but does not provide actionable ranking signal for personalization.

#### 2. Conclusion
- **DROP:** Do not implement manufacturer-level fallback features. The engine should transition from **Item-level** directly to **Brand-level** or **Category-level** signals, skipping the manufacturer layer entirely to avoid injecting 20% more noise into the predictions.

#### 3. Future Exploration (Sub-Idea)
- **Manufacturer-Category Specialization:** Does a manufacturer's stability improve if we only look at their 'hero' category (e.g., Abbott only in Milk)?

### Idea 34: Size Ladder by Age-Proxy
**Usefulness Score:** 0.90
**Status:** KEEP

#### 1. Empirical Findings
- **Diaper Precision:** Diapers show the most systematic ladder: **82.3%** of customers buying Age 0.6Y (M) move to 1.2Y (L).
- **Fashion Applicability:** Fashion also shows high upgrade rates: **61.3%** move from 0.0Y (NB) to 0.125Y (0-3M), and **59.9%** move from 0.75Y (9M) to 1.0Y (12M).
- **Predictive Velocity:** Transitions are non-random and follow a strictly increasing age-proxy trajectory, confirming that size is a reliable proxy for child growth stage.

#### 2. Conclusion
- **KEEP:** Implement the "Next-Size-Up" predictive engine. When a customer reaches the end of their predicted "Time-in-Size" (based on purchase frequency), boost items in the next age bucket by **1.5x**.

### Idea 35: Price Tier Specialization (Dual Track)
**Usefulness Score:** 0.85
**Status:** KEEP

#### 1. Empirical Findings
- **Absolute Specialization:** Named brands have a **Median Entropy of 0.0**, meaning the vast majority of identified brands exist in **only one price tier** (Budget, Standard, or Premium).
- **Unknown Profile:** The 'Không xác định' bucket is significantly skewed toward **Budget (49.8%)**, but still contains 27% Premium items, making it a "Budget-leaning generalist" rather than a pure budget proxy.
- **Signal Strength:** This creates a powerful filtering signal: once a customer's price affinity is identified, we can safely boost/penalize hundreds of brands simultaneously.

#### 2. Conclusion
- **KEEP:** Store `brand_dominant_tier` as a feature. Implement **Price Affinity Filtering**: calculate a customer's tier distribution and apply a **0.5x penalty** to items from brands that strictly belong to tiers the customer never buys from.

### Idea 36: Location Assortment Coverage
**Usefulness Score:** 0.90
**Status:** KEEP

#### 1. Empirical Findings
- **Extreme Catalog Fragmentation**: The average location only stocks **10.0%** of the global catalog. Even the largest stores only reach **27.8%** coverage.
- **Item Sparsity**: **12.4%** of items are "Hyper-Local," appearing in only a single location.
- **Category Specialization**: Locations like 'HCM - Aeon Mall Tân Phú' are heavily skewed toward 'Thời trang' (53.5% of unique items), indicating that stock is not just limited by volume, but specialized by store format.

#### 2. Conclusion
- **KEEP**: Implementing a **Location-Aware Filter** is mandatory. Global "Hot Items" should be penalized or masked if they have no historical sales footprint at the user's primary location.

#### 3. Actionable Insight
- **Feature Engineering**: Store a `location_item_whitelist` (Bloom filter or sparse bitmask) for each location.
- **Ranking Logic**: Apply a **0.1x penalty** (or hard mask) to items not in the location's historical assortment. This ensures the engine doesn't recommend "Ghost Items" that the customer cannot physically purchase.

### Idea 37: Location-Specific Category Mix
**Usefulness Score:** 0.85
**Status:** KEEP

#### 1. Empirical Findings
- **High Stability**: Correlation of **0.948** between H1 and H2 category lifts across locations. Store "personalities" are persistent and not driven by seasonal noise.
- **Extreme Specialization**:
    - **Fashion Hubs**: Mall locations (e.g., Aeon Mall Tân Phú) have a **6.1x lift** for Fashion.
    - **Toy Hubs**: Aeon Mall Hà Đông has a **5.2x lift** for Toys.
    - **Textile Hubs**: Specific rural/street locations show **4.6x lift** for Textiles.
- **Administrative Outliers**: Central warehouses show extreme lifts (158x) for non-item categories like "Gói Hội Viên," which should be filtered out.

#### 2. Conclusion
- **KEEP**: Location-category lift is the single best signal for **Cold-Start Personalization**. When a user's history is missing, the engine should default to the "Store Personality" rather than the "Global Average."

#### 3. Actionable Insight
- **Feature Engineering**: Store a `location_category_lift` vector for each store.
- **Ranking Logic**: For users at a specific location, boost items in high-lift categories by **1.2x**. This aligns recommendations with local stock depth and store-specific shopper profiles.

### Idea 38: Location-Item Availability Gap
**Usefulness Score:** 0.90
**Status:** KEEP

#### 1. Empirical Findings
- **Systemic Sparsity**: The median **Category Gap Rate** is **96.3%**. For a typical item, it is absent from 96% of the stores that sell its parent category.
- **Fragmentation by Category**:
    - **Fashion & Toys**: Extremely fragmented (**90%+ gap**). Global popularity is a "Ghost Signal" here; items are highly localized.
    - **Milk & Baby Food**: Significantly more uniform (**38-47% gap**). Global popularity is a much safer signal for these "Essential" categories.
- **Presence vs. Volume**: Top 10 global items (mostly Milk) have nearly 100% presence, but this drops off exponentially as we move down the tail.

#### 2. Conclusion
- **KEEP**: Implementing an **Item-Level Location Filter** is mandatory for Fashion, Toys, and Home Care. Milk can use a more relaxed "Global" signal.

#### 3. Actionable Insight
- **Feature Engineering**: Store a `high_sparsity_category` flag.
- **Ranking Logic**: For items in high-sparsity categories (Fashion, Toys), apply a **Hard Mask** if they have 0 sales at the user's primary location. For low-sparsity categories (Milk), use a **0.5x penalty** instead of a hard mask to allow for potential stock expansion.

### Idea 39: Category-Lifecycle Interaction
**Usefulness Score:** 0.80
**Status:** KEEP

#### 1. Empirical Findings
- **High Churn Categories**: **Fashion (Thời trang)** and **Accessories (Phụ kiện)** show the highest **Decline rates (44-46%)**, confirming rapid seasonal SKU turnover.
- **Stable Categories**: **Textile** is the most robust, with **70%** of items classified as Stable.
- **Trend Volatility**: Surprisingly, **Diapers (Tã)** and **Milk (Sữa nước)** show the highest trend volatility (**0.23 std**). This suggests that demand for specific SKUs in these categories is highly sensitive to external factors (promotions, stock-outs), creating sharp trend spikes.
- **Look-back Window Sensitivity**: Categories with high decline rates require a more reactive ranking system to avoid recommending stale SKUs.

#### 2. Conclusion
- **KEEP**: Differing volatility and stability profiles prove that a **Category-Specific Trend Window** is required.

#### 3. Actionable Insight
- **Feature Engineering**: Store `category_lifecycle_volatility` as a metadata field.
- **Ranking Logic**:
    - **Fashion/Accessories**: Use a **30-day Trend Window** to catch new arrivals and prune dying trends immediately.
    - **Textile/Milk**: Use a **120-day Trend Window** to filter out short-term promotional "blips" and focus on long-term brand momentum.

### Idea 40: Item Concentration vs Breadth
**Usefulness Score:** 0.75
**Status:** KEEP

#### 1. Empirical Findings
- **Archetype 1: Global Staples (9,099 items)**: High volume and broad location presence. These are the core discovery drivers for the engine.
- **Archetype 2: Niche Heroes (5,973 items)**: Extremely high customer concentration (HHI) but very narrow location breadth. These are high-loyalty items that are invisible to the general market.
- **Archetype 3: Local Staples (4,781 items)**: Dispersed across few locations. These are likely region-specific must-haves.
- **Inversed Correlation**: A moderate negative correlation (-0.33) confirms that the more widely an item is available, the less its demand is concentrated in a few hands.

#### 2. Conclusion
- **KEEP**: The ranking engine must be "Archetype-Aware." Recommending a "Niche Hero" to a random user is low-utility "noise," but missing it for its core loyalists is a major conversion loss.

#### 3. Actionable Insight
- **Feature Engineering**: Tag items with their `item_archetype`.
- **Ranking Logic**:
    - **Niche Heroes**: Apply a **0.7x penalty** for general users, but a **1.5x boost** if the user has previously bought from that niche or is at a high-concentration location for that item.
    - **Global Staples**: Use as the "Discovery Baseline" for cold-start and exploration slots.

### Idea 41: Customer Lifecycle Stage
**Usefulness Score:** 0.85
**Status:** KEEP

#### 1. Empirical Findings
- **The Discovery Mindset**: **New customers** exhibit the highest **Category Entropy (3.52)**. They are in a state of high exploration as they discover the assortment.
- **The Specialization Drift**: As customers become **Active**, their entropy drops to **3.16**. They "Narrow Down" their focus to specific staple categories and brands.
- **The Retention Gap**: A massive segment of **1.3M+ users** are currently **Dormant or Churned**. Their entropy (3.3-3.4) is higher than Active users, suggesting they haven't settled into staples yet.
- **Quantity Variance**: New users (1.68) and Active users (1.62) buy significantly more items per transaction than Churned users (1.42).

#### 2. Conclusion
- **KEEP**: The ranking engine must distinguish between "Explorers" (New) and "Specialists" (Active). Furthermore, "Reactivation" for Dormant/Churned users requires a focus on high-repeat staples rather than exploration.

#### 3. Actionable Insight
- **Feature Engineering**: Store `customer_lifecycle_stage` as a primary user feature.
- **Ranking Logic**:
    - **Active Users**: Focus on **Repetition and Precision**. Boost their most-frequent brands and categories.
    - **New Users**: Focus on **Discovery**. Increase the "Exploration Temperature" and show a diverse mix of Global Staples and High-Trend items.
    - **Dormant/Churned**: Focus on **Reactivation**. Rank their historical top-3 items at #1, #2, and #3 to reduce friction during their return.

### Idea 42: Category Affinity by Customer
**Usefulness Score:** 0.90
**Status:** KEEP

#### 1. Empirical Findings
- **Extreme Specialization**: **82.2%** of all customers have a Category HHI **> 0.3**, indicating a heavy reliance on a single "Anchor Category."
- **Narrow vs Broad**: 
    - **Narrow Shoppers (1,663,165 users)**: The vast majority. These users buy almost exclusively from 1-2 categories.
    - **Broad Shoppers (198,961 users)**: A small but highly active segment that explores the full catalog.
- **Volume Correlation**: Category Entropy has a **0.51 correlation** with total transaction count. This proves that as customers become "Power Users," they naturally expand their category horizons.
- **Tenure Drift**: HHI has a **-0.30 correlation** with tenure, meaning the longer a customer stays, the less concentrated their category mix becomes.

#### 2. Conclusion
- **KEEP**: The ranking engine must respect the "Anchor Category" for the 82% specialized majority to avoid irrelevant "noise." However, it should dynamically increase discovery temperature as a user's transaction volume grows.

#### 3. Actionable Insight
- **Feature Engineering**: Tag users with their `shopping_type` (Narrow/Balanced/Broad) and `primary_anchor_category`.
- **Ranking Logic**:
    - **Narrow Shoppers**: Apply a **2.0x boost** to their `primary_anchor_category` and a **0.5x penalty** to unrelated categories. Keep them in their comfort zone.
    - **Broad Shoppers**: Use a **Neutral** weight for category-affinity, allowing the global trend and item-level similarity to drive cross-category exploration.

### Idea 43: Brand Loyalty Within Preferred Categories
**Usefulness Score:** 0.85
**Status:** KEEP

#### 1. Empirical Findings
- **The Loyalty Premium**: Brand HHI is significantly higher **Inside (0.59)** compared to **Outside (0.46)** a user's top-2 categories.
- **Repeat Rate Spike**: Customers repeat brands **40%** of the time in their preferred categories, but only **31%** when shopping elsewhere.
- **Top Brand Share**: The average share of the #1 brand is **64%** in preferred categories vs. **52%** outside.
- **Significance**: The HHI difference of **0.127** clearly passes the 0.1 decision threshold.

#### 2. Conclusion
- **KEEP**: Brand preference is not a global user trait but is highly category-dependent. Users have "Anchor Brands" in their favorite categories but act like explorers in others.

#### 3. Actionable Insight
- **Ranking Logic**:
    - **Inside Preferred Category**: Apply a **1.5x Brand Loyalty Boost**. Users are extremely likely to stick to their established brand habits in these areas.
    - **Outside Preferred Category**: Use **Neutral Brand Weights**. Prioritize **Global Popularity** and **Discovery** features to help the user find their next anchor brand.

### Idea 44: Customer-Item Repeat Patterns
**Usefulness Score:** 0.95
**Status:** KEEP

#### 1. Empirical Findings
- **The Replenishment King**: **Milk (Sữa)** has a **32% repeat probability** with a median gap of **13 days**. This is the highest replenishment signal in the catalog.
- **The Diaper Cycle**: **Diapers (Tã)** have a **19% repeat probability**.
- **One-Off Categories**: **Fashion (Thời trang)** and **Toys (Đồ chơi)** have a median gap of **0.0 days**. Users rarely buy the exact same SKU twice in these categories.
- **Subscription Cycles**: **Membership (Gói Hội Viên)** exhibits a **132-day** cycle, matching quarterly billing patterns.
- **Price Sensitivity**: Repeat probability is highest for **Standard (mid-range)** items, while Budget items have lower repeat rates, possibly indicating trial-and-error behavior.

#### 2. Conclusion
- **KEEP**: Predictable repeat cycles for core consumables allow for a powerful replenishment-aware ranking engine. For discretionary categories, the absence of repeats should trigger a "diversity boost" to avoid fatigue.

#### 3. Actionable Insight
- **Feature Engineering**: Calculate `item_median_repeat_gap` and `days_since_last_purchase` for every user-item pair.
- **Ranking Logic**:
    - **Replenishment Categories (Milk/Food/Diapers)**: If `days_since_last_purchase` is within **+/- 3 days** of the `item_median_repeat_gap`, apply a **3.0x Replenishment Boost**.
    - **Discretionary Categories (Fashion/Toys)**: Apply a **0.1x Penalty** to items already purchased by the user (avoid repetition of non-consumables).

### Idea 45: Size Progression Behavior
**Usefulness Score:** 0.80
**Status:** KEEP

#### 1. Empirical Findings
- **High-Progression Categories**: **Accessories (Phụ kiện)** exhibit a **43% upgrade rate**, indicating rapid growth or sequential purchasing (Size 1 -> 2 -> 3).
- **The Stability Zone**: **Diapers (Tã)** have an **87% stability rate**. Users stay in the same weight bracket for long periods, with only **5.8%** upgrading in the observed 2025 window.
- **Directional Drift**: Across all size-mapped items, the **Upgrade Rate (9.3%)** is 2.5x higher than the **Downgrade Rate (3.8%)**, confirming a sequential "Size Ladder."
- **Tenure Correlation**: Veteran customers (Tenure > 180d) show a clear drift toward larger average sizes compared to new users.

#### 2. Conclusion
- **KEEP**: Size-aware recommendations should be **category-specific**. In Accessories and Fashion, we should anticipate the upgrade. In Diapers, we should prioritize stability until a "Size Fatigue" threshold is reached.

#### 3. Actionable Insight
- **Ranking Logic**:
    - **High-Progression (Accessories/Fashion)**: If a user has purchased the same size **3 times**, apply a **1.5x Boost** to the **Next Size Up** in the age-proxy ladder.
    - **Stability Categories (Diapers)**: Default to the **Current Size**. Trigger an upgrade recommendation *only* if the user's tenure in that size exceeds the `item_lifecycle_stage` duration.

### Idea 46: Customer Segment Clustering (RFM-style)
**Usefulness Score:** 0.90
**Status:** KEEP

#### 1. Empirical Findings
- **Segment 3 (Whales / Super-Explorers)**: 172k customers with extreme activity. Average frequency of **40 purchases** and **173 units**. Very high diversity (**Entropy 2.45**) and low concentration (**HHI 0.27**).
- **Segment 1 (Active Discoverers)**: 1.25M customers (the core base). Moderate frequency (6.6) and high diversity (**Entropy 1.75**). They enjoy cross-category browsing.
- **Segment 2 (Targeted Habituals)**: 703k customers. Extremely high concentration (**HHI 0.97**) and near-zero discovery (**Entropy 0.05**). They buy from exactly one category.
- **Segment 0 (Hibernators)**: 693k customers. Long recency (**250 days**). Primarily single-category shoppers (HHI 0.82) who haven't returned.

#### 2. Conclusion
- **KEEP**: These segments provide the "Persona Context" for the ranking engine. Universal weights are ineffective when one segment wants 100% discovery and another wants 100% habit.

#### 3. Actionable Insight
- **Segment-Aware Ranking**:
    - **Whales & Discoverers**: Apply a **2.0x Discovery Boost**. Prioritize cross-category exploration and "New Arrival" signals.
    - **Targeted Habituals**: Apply a **5.0x Loyalty Penalty** to any items outside their established `category_l1`. Don't distract them; keep them in their habit.
    - **Hibernators**: Ignore personalization (which may be stale). Use **Global Popularity** and **Trending** signals to lure them back.

### Idea 47: Location Affinity by Customer
**Usefulness Score:** 0.85
**Status:** KEEP

#### 1. Empirical Findings
- **Extreme Hub Concentration**: **85.1%** of customers have a Location HHI > 0.5. 
- **The "Single Store" Majority**: The median Location HHI is **1.0**, meaning more than half of all customers shop at a single location exclusively.
- **Low Mobility**: Only **17.2%** of customers have shopped at 3 or more distinct locations.
- **Expansion with Value**: There is a **-0.35 correlation** between visit frequency and location HHI. As customers shop more, they tend to use slightly more locations (expanding their hub), but still remain highly concentrated.
- **Top Share**: Average share of the #1 location is **88%**.

#### 2. Conclusion
- **KEEP**: Customers are geographically stationary. A global recommendation for an item that is out of stock at their specific "Home Hub" is a wasted slot.

#### 3. Actionable Insight
- **Ranking Logic**:
    - **Stationary Shoppers (HHI > 0.5)**: Apply a **Location Availability Filter**. Items out of stock at their `primary_location_hub` receive a **0.2x Rank Penalty**.
    - **Mobile Shoppers (HHI < 0.5)**: Use **Regional Stock** (e.g., city-level) rather than hub-specific. They have shown a willingness to travel between locations.

### Idea 48: Price Tier Movement by Customer Over Time
**Usefulness Score:** 0.80
**Status:** KEEP

#### 1. Empirical Findings
- **High Economic Mobility**: **71.0%** of multi-purchase customers show a predictable price tier drift over their tenure. 
- **The Bifurcation of Taste**: The base is split perfectly between **Trading Up (35.5%)** and **Trading Down (35.5%)**.
- **Tier Consistency**: Only **29.0%** of customers remain within their initial price tier (Budget/Standard/Premium) throughout their entire lifecycle.
- **Up-Trading Archetype**: These users typically start with Budget staples (Milk/Diapers) and drift toward Premium Fashion or Accessories.
- **Down-Trading Archetype**: These users often start with trial purchases in mid-tier and stabilize into high-volume Budget replenishment cycles.

#### 2. Conclusion
- **KEEP**: A static "Price Sensitivity" score is insufficient. The ranking engine must use the **Direction of Drift** to determine if the user is currently seeking value or quality.

#### 3. Actionable Insight
- **Ranking Logic**:
    - **Up-Traders (Drift > 0.1)**: Apply a **1.2x Boost** to **Premium Tier** items in the target category. Prioritize brand reputation and quality signals.
    - **Down-Traders (Drift < -0.1)**: Apply a **1.3x Boost** to **Budget Tier** items. Prioritize unit-price efficiency and bulk-discount labels.
    - **Stable Shoppers**: Maintain strict adherence to their **Current Tier** (Standard filter).

### Idea 49: Purchase Velocity and Seasonality by Customer
**Usefulness Score:** 0.85
**Status:** KEEP

#### 1. Empirical Findings
- **The Core Pulse**: The median customer velocity is **2.0 purchases per month**. 
- **The High-Velocity Whales**: The top 25% of the base shop **4+ times per month**, creating a constant need for catalog "Freshness."
- **Low Individual Seasonality**: Only **7.0%** of customers show extreme seasonal variance (CV > 1.0). This indicates that the majority of active users are **Consistent Habitualists** who shop at a steady rhythm year-round.
- **The November Super-Peak**: November (Month 11) sees a massive spike in activity (**336k peak customers**), likely driven by 11.11 or Black Friday promotions.
- **December Drop-off**: December (Month 12) shows a sharp decline (8k), potentially due to data windowing or a post-November saturation effect.

#### 2. Conclusion
- **KEEP**: Velocity is a high-leverage ranking feature. High-velocity users exhaust personalized recommendations quickly and need faster injection of "New Arrival" or "Trending" items.

#### 3. Actionable Insight
- **Ranking Logic**:
    - **High-Velocity Shoppers (Velocity > 4/mo)**: Apply a **1.5x Freshness Boost**. Prioritize items that have NOT been shown to the user in the last 3 visits.
    - **Seasonal Peakers (CV > 1.0)**: During their **Predicted Peak Month**, apply a **2.0x Boost** to their top-HHI categories (stock-up prediction).
    - **Steady Shoppers**: Prioritize **Replenishment Cycles** (Idea 44) over discovery.

### Idea 50: Cross-Category Purchase Patterns by Customer Segment
**Usefulness Score:** 0.90
**Status:** KEEP

#### 1. Empirical Findings
- **Distinct Segment DNAs**: The Jaccard similarity of top category preferences between "Explorers" and "Loyalists" is only **0.538**. This proves that a one-size-fits-all bundle strategy is inefficient.
- **The Membership Anchor**: Explorers (Whales) show an extreme **2.28x Lift** for bundles involving **Gói Hội Viên (Membership)** and high-turnover categories like **Sanitation (Vệ sinh)** and **Family Food**.
- **Loyalist Staples**: Loyalists are defined by high-volume staples (Milk/Baby Food). However, these bundles have **Low Lift (< 0.7)** relative to the global average because they are universally popular. Loyalists represent the "Global Baseline" of consumption.
- **Discovery Pathing**: Explorers are **2.3x more likely** to bundle diverse categories like **Textile** and **Accessories** than the average user.

#### 2. Conclusion
- **KEEP**: Cross-selling must be segment-aware. We should use the "Membership Anchor" to drive discovery for Whales and focus on "High-Volume Replenishment" for Loyalists.

#### 3. Actionable Insight
- **Ranking Logic**:
    - **Explorers (Segment-Aware Cross-Sell)**: If the user is an Explorer and views a membership item, boost categories with **Lift > 1.3** (Sanitation, Family Food, Textile) by **1.5x**.
    - **Loyalists**: Stick to the **Global Staple Bundles** (Idea 44) but prioritize **Price/Bulk** efficiency.
    - **Cross-Persona Bridge**: Use the 53.8% category overlap to identify "Gateway Items" that can migrate a Habitual into an Explorer.















