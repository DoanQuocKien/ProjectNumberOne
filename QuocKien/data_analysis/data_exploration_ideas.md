# Data Exploration Ideas for PIR

This document is the concept stage for the QuocKien exploration work. It is intentionally descriptive and statistical, not an implementation yet. The later execution phase should build one Jupyter notebook section per idea, using Polars for all data work.

## Scope

- Core datasets: `transaction_full_2025.parquet` and `items.parquet`.
- Excluded dataset: `event_full_2025.parquet`.
- Tooling: Jupyter notebook plus Polars.
- Goal: test whether each idea has enough signal to justify a later feature, visualization, or recommendation strategy.

## Shared Notebook Rules

1. Use Polars for loading, joining, filtering, grouping, and statistical aggregation.
2. Keep every idea in its own notebook block with the same structure:
   - hypothesis
   - datasets used
   - columns to inspect
   - transformation steps
   - statistical formula or test
   - visualization plan
   - decision rule
3. Prefer math-based conclusions over descriptive commentary.
4. Use the same base time fields everywhere: day, week, month, weekday, rolling windows.
5. Do not build model features or recommender changes yet.

## Idea 1: Holiday Effect in Vietnam

### Question
Do major Vietnamese holidays change sales in the week before, during, and after the holiday?

### Datasets
- `transaction_full_2025.parquet`

### Columns to Inspect
- `updated_date`
- `quantity`
- `customer_id`
- `item_id`

### Holidays to Test
- Tet Nguyen Dan
- Hung Kings Commemoration Day
- Reunification Day
- International Labor Day
- National Day
- Mid-Autumn Festival

### Method
- Convert transaction timestamps to daily sales totals.
- Define three windows for each holiday: pre-holiday, holiday, and post-holiday.
- Choose a baseline window from nearby non-holiday days.

### Statistics
- Holiday lift:
  - `lift = (S_holiday - S_baseline) / S_baseline`
- Compare daily sales between holiday windows and baseline using:
  - two-sample t-test if the distribution is close to normal
  - Mann-Whitney U if sales are skewed
- If counts are very discrete or skewed, fit a Poisson or negative binomial regression with holiday-window indicators.

### Visualizations
- Daily sales line chart with shaded holiday windows
- Bar chart of holiday lift by holiday
- Heatmap for weekday versus holiday-window sales

### Decision Rule
- Keep the idea if multiple holidays show consistent and statistically meaningful demand shifts.

## Idea 2: Brand Commitment and Manufacturer Commitment

### Question
Do users concentrate purchases in one or two brands? Does the same pattern apply to manufacturers?

### Datasets
- `transaction_full_2025.parquet`
- `items.parquet`

### Columns to Inspect
- `customer_id`
- `item_id`
- `brand`
- `manufacturer`
- `updated_date`
- `quantity`

### Method
- Join transaction rows to item metadata.
- For each customer, calculate the distribution of purchases across brands and manufacturers.

### Statistics
- Herfindahl-Hirschman Index:
  - `HHI = sum(p_i^2)` where `p_i` is the share of purchases in brand or manufacturer `i`
- Entropy:
  - `H = -sum(p_i log(p_i))`
- Top-1 share and top-2 share
- Repeat-rate by brand and manufacturer
- Compare against a random or shuffled baseline if needed.

### Visualizations
- Histogram of HHI
- Cumulative share curve
- Bar charts of top brands and top manufacturers
- Scatter plot of concentration versus purchase volume

### Decision Rule
- Keep the idea if brand or manufacturer loyalty is strong enough to help ranking or user segmentation.

## Idea 3: Category Connections and Co-Purchase Structure

### Question
Are items in the same category or in related categories bought together more often than chance?

### Datasets
- `transaction_full_2025.parquet`
- `items.parquet`

### Columns to Inspect
- `customer_id`
- `item_id`
- `category`
- `category_l1`
- `category_l2`
- `category_l3`
- `brand`
- `manufacturer`
- `updated_date`

### Method
- Join transaction data to item metadata.
- Build baskets at the customer-day or customer-transaction level.
- Count category pairs and item pairs.

### Statistics
- Support:
  - `support(A, B) = baskets containing both A and B / total baskets`
- Confidence:
  - `confidence(A -> B) = support(A, B) / support(A)`
- Lift:
  - `lift(A -> B) = confidence(A -> B) / support(B)`
- Jaccard similarity for pair strength
- Pointwise mutual information for ranking

### Visualizations
- Weighted network graph of category links
- Heatmap of category-to-category lift
- Sankey or chord diagram for strong category flows

### Decision Rule
- Keep the idea if some category pairs have lift clearly above 1 and remain stable across users or time.

## Idea 4: Size Normalization and Rebuy-Up Behavior

### Question
Can size be normalized into standard age buckets, and do customers rebuy similar items in larger sizes as children grow?

### Datasets
- `transaction_full_2025.parquet`
- `items.parquet`

### Columns to Inspect
- `size`
- `item_id`
- `customer_id`
- `updated_date`
- `quantity`
- category fields

### Method
- Parse size labels into standardized buckets such as NB, 3M, 6M, 9M, and 12M.
- Track repeated purchases of the same item or category by customer.

### Statistics
- Size transition matrix from smaller to larger buckets
- Upgrade rate:
  - `upgrade_rate = upgrade_sequences / repeat_sequences`
- Median time gap between repeated purchases
- If direct child-age labels are unavailable, use repeated purchase sequences by customer as the sequence unit.

### Visualizations
- Size transition heatmap
- Bar chart of rebuy-up frequency by category
- Example customer sequence plot

### Decision Rule
- Keep the idea if size transitions are systematic enough to justify size-aware recommendations or replenishment logic.

## Idea 5: Purchase Seasonality and Weekday Pattern

### Question
Do sales follow repeatable weekday and monthly seasonality beyond holiday effects?

### Datasets
- `transaction_full_2025.parquet`

### Columns to Inspect
- `updated_date`
- `quantity`
- `customer_id`
- `item_id`

### Method
- Aggregate sales by day, weekday, week, and month.
- Compare periods across the calendar.

### Statistics
- Seasonal indices by weekday and month
- ANOVA across weekday groups if assumptions are reasonable
- Kruskal-Wallis if distributions are skewed
- Z-score outlier detection for unusually high or low days

### Visualizations
- Daily sales trend line
- Weekday boxplots
- Month-by-weekday heatmap

### Decision Rule
- Keep the idea if regular seasonality is strong enough to support timing features or demand planning.

## Idea 6: Replenishment and Repeat-Purchase Cycle

### Question
Are consumables and baby-care staples repurchased on a predictable cycle?

### Datasets
- `transaction_full_2025.parquet`

### Columns to Inspect
- `customer_id`
- `item_id`
- `updated_date`
- `quantity`

### Method
- For each customer-item pair, compute the time gap between repeated purchases.
- Repeat the same logic for category-level replenishment if item-level repeats are sparse.

### Statistics
- Median interpurchase gap
- Interquartile range of gaps
- Repeat probability within 7, 14, and 30 days
- Survival-style summaries or Kaplan-Meier curves if repeat events are frequent enough
- Optional hazard-style summary by category

### Visualizations
- Histogram of time gaps
- Survival curve
- Repeat-rate by category

### Decision Rule
- Keep the idea if the repeat cycle is sharp enough to support replenishment reminders or next-buy predictions.

## Idea 7: Basket Structure and Association Rules

### Question
Which items or categories appear as stable bundles in the same basket or close time window?

### Datasets
- `transaction_full_2025.parquet`
- `items.parquet`

### Columns to Inspect
- `customer_id`
- `updated_date`
- `item_id`
- category fields
- `brand`
- `manufacturer`

### Method
- Define baskets at a consistent level, such as customer-day.
- Run association-rule style analysis on item pairs and category pairs.

### Statistics
- Support, confidence, and lift for rule ranking
- Minimum support threshold to remove noise
- Optional shuffled baseline to confirm the rules are not random

### Visualizations
- Network graph of the strongest rules
- Ranked rule table
- Clustered adjacency heatmap

### Decision Rule
- Keep the idea if bundle patterns have enough support to matter operationally.

## Idea 8: Customer Concentration Versus Diversity

### Question
Are some customers narrow-repeat buyers while others are broad explorers?

### Datasets
- `transaction_full_2025.parquet`
- `items.parquet`

### Columns to Inspect
- `customer_id`
- category fields
- `brand`
- `manufacturer`
- `updated_date`
- `quantity`

### Method
- For each customer, compute diversity and concentration metrics across categories, brands, and manufacturers.

### Statistics
- Category entropy
- Brand entropy
- Manufacturer entropy
- Distinct-item count
- Concentration indices such as HHI or Gini-style summary
- Compare user groups with quantiles or clustering
- Use ANOVA or Kruskal-Wallis to compare spending or repeat behavior across segments

### Visualizations
- Scatter plot of diversity versus purchase volume
- Boxplots by user segment
- Compact segment summary panel

### Decision Rule
- Keep the idea if user diversity segments are clearly separated and stable enough to support personalized ranking strategies.

## Idea 9: Item Lifecycle: New, Growth, Stable, Decline

### Question
Do items follow lifecycle stages that affect recommendation priority?

### Datasets
- `transaction_full_2025.parquet`
- `items.parquet`

### Columns to Inspect
- `item_id`
- `updated_date`
- `quantity`
- `category`
- `brand`
- `manufacturer`

### Method
- Aggregate monthly sales per item.
- Measure trend, growth, volatility, and peak timing.

### Statistics
- Moving-average slope
- Growth rate
- Volatility
- Peak month timing
- Simple threshold-based or change-point-style grouping into new, rising, stable, and declining items

### Visualizations
- Item lifecycle curves
- Slope histogram
- Category-level trend panels

### Decision Rule
- Keep the idea if lifecycle stage is useful for popularity forecasting or cold-start ranking.

## Idea 10: Price Sensitivity and Basket Price Consistency
### Question
- Do customers tend to buy items within a similar price tier, and how do locational price deviations impact purchase behavior?

### Datasets
- `transaction_full_2025.parquet`
- `items.parquet`

### Columns to Inspect
- `customer_id`
- `item_id`
- `price`
- `location_name` 
- `updated_date`

### Method
- Join `transaction_full_2025.parquet` with `items.parquet` to establish the baseline price for each product.
- Group by `item_id` and `location_name` to identify price deviations (anomalies from the national/global mode price).
- Assign items to price tiers (e.g., budget, standard, premium) within their respective categories using quantiles.
- Calculate the distribution of price tiers for each customer's historical purchases and within individual baskets.

### Statistics
- Coefficient of Variation (CV):
Measure the variance of price tiers within a customer's basket compared to the global variance.
- Intraclass Correlation Coefficient (ICC):
ICC to measure how strongly items in the same basket resemble each other in terms of price tier.
- Locational Deviation Delta:
price_delta = (P_location - P_mode) / P_mode
- Elasticity proxy: compare sales volume for the same item between locations with baseline prices vs. locations with deviated prices (using an independent t-test or Mann-Whitney U).

### Visualizations
- Boxplots of unit prices for top items grouped by location to highlight deviations.
- Stacked bar chart showing the proportion of budget/standard/premium items per customer segment.
- Scatter plot of basket size versus average basket price-tier variance.

### Decision Rule
Keep the idea if customers show a strong, statistically significant affinity for specific price tiers, or if locational deviations provide enough signal to adjust local ranking algorithms.

## Recommendation-System-Focused Interactions Bank

These 20 ideas focus on **item-customer-location-category interactions** that directly drive ranking decisions. They are intentionally narrower and designed to extract personalized signals for each customer, not aggregate demand patterns.

### Suggested target

> [!NOTE]
> Columns `brand` and `manufacturer` contain "Không xác định" (Unknown) values. For manufacturers, this is the majority value (~80-90%). Analysis must either handle this as a separate segment or filter it out to avoid aggregate bias.

- All 20 ideas should be analyzed; this is the core interaction layer for recommendations.
- Focus is entirely on item selection, not timing or demand forecasting.
- These ideas work best in combination with customer behavior (41–50) and item structure (31–40).

### Idea 11: Item Popularity by Customer Segment

#### Question
Which items are popular within each customer segment (RFM-based or lifecycle-based), and how does popularity vary across segments?

#### Datasets
- `transaction_full_2025.parquet`
- `items.parquet`

#### Columns to Inspect
- `customer_id`
- `item_id`
- `quantity`
- `updated_date`
- `category_l1`
- `category_l2`
- `category_l3`

#### Method
- Segment customers using RFM (Recency, Frequency, Monetary) or lifecycle stage classification.
- For each segment, compute item purchase frequency, average quantity, and total quantity purchased.
- Rank items per segment and identify segment-specific top-20 items.
- Calculate affinity lift: (segment item purchase rate) / (global item purchase rate).

#### Statistics
- Top-20 items per segment with purchase counts
- Segment-item affinity lift (items with lift > 1.5 are segment-specific hits)
- Segment-specific item diversity (count of items purchased by segment)
- Coefficient of variation of item popularity across segments (higher CV = more segment differentiation)

#### Visualizations
- Grouped bar chart: top 15 items by segment, showing purchase frequency per segment
- Heatmap: items (rows) × segments (columns), with affinity lift as color intensity
- Scatter plot: segment size (x-axis) vs. average top-item lift (y-axis) to show segment differentiation power
- Venn diagram or overlap plot showing which items are unique to segments vs. shared

#### Decision Rule
- Keep if top items differ meaningfully across customer segments (e.g., >30% of top items in segment A are not in top 20 of segment B) or if average affinity lift for segment-specific items exceeds 1.5.

### Idea 12: Category-Customer Affinity Stability

#### Question
Is a customer's category preference stable over time, or does it drift significantly?

#### Datasets
- `transaction_full_2025.parquet`
- `items.parquet`

#### Columns to Inspect
- `customer_id`
- `item_id`
- `updated_date`
- `category_l1`
- `category_l2`
- `quantity`

#### Method
- Split each customer's transaction history into two equal time periods (first half and second half by date).
- For each period, compute category share distribution: (purchases in category / total purchases).
- Calculate rank-order correlation of category shares between periods.
- Measure category churn: proportion of top-5 categories in period 1 that disappear from top-5 in period 2.

#### Statistics
- Spearman rank correlation of category shares between periods (μ and σ across all customers)
- Kendall tau (alternative rank correlation measure)
- Category churn rate (% of top-5 categories that fall out of top-5 in period 2)
- Median category share stability: correlation for top 1, 2, 3, 5 categories separately
- Kullback-Leibler divergence of category distributions between periods

#### Visualizations
- Scatter plot: category share in period 1 (x-axis) vs. period 2 (y-axis) with 45° reference line
- Boxplot: rank-correlation distribution across customers
- Stacked bar chart: top 10 categories, showing before/after share for 5 representative stable and 5 unstable customers
- Heatmap: category (rows) × stability decile (columns), showing transition between stable/unstable customers

#### Decision Rule
- Keep if median rank correlation > 0.6 or if category churn rate < 20%, indicating category preferences are reasonably stable for ranking personalization.

### Idea 13: Customer-Item Discovery Path

#### Question
How do customers discover new items—primarily through categories they already buy, or by actively exploring new categories?

#### Datasets
- `transaction_full_2025.parquet`
- `items.parquet`

#### Columns to Inspect
- `customer_id`
- `item_id`
- `updated_date`
- `category_l2`
- `category_l3`

#### Method
- For each customer, identify the first purchase of each item.
- For each first-purchase item, compare its category_l2 to the customer's category distribution in all prior purchases.
- Classify discoveries as "in-preferred" (category was already in customer's top 3 categories) or "exploratory" (new category).
- Measure discovery bias by customer segment (RFM-based).

#### Statistics
- Proportion of in-category vs. exploratory discoveries (overall and by segment)
- Cross-category discovery rate: P(first item in category C_new | customer history in categories C_1, C_2, …)
- Discovery entropy by customer segment
- Lift of discovery-in-preferred vs. random selection baseline
- Repeat rate after discovery for in-category vs. exploratory items

#### Visualizations
- Pie chart: proportion of in-preferred vs. exploratory discoveries
- Stacked bar chart: discovery type distribution by customer RFM segment
- Sankey diagram: customer category history → first-time item category (showing flow intensity)
- Scatter plot: customer exploration entropy (x-axis) vs. repeat rate on new items (y-axis)

#### Decision Rule
- Keep if either (1) in-category discoveries have lift > 1.2 compared to random, OR (2) exploratory discoveries repeat at rates < 50% of in-category discoveries, indicating different ranking strategies may apply.

### Idea 14: Item Affinity by Price Tier Within Category

#### Question
Within each category and price tier, which items are most preferred by each customer segment, and does tier constrain item choice?

#### Datasets
- `transaction_full_2025.parquet`
- `items.parquet`

#### Columns to Inspect
- `customer_id`
- `item_id`
- `price`
- `quantity`
- `category_l2`
- `category_l3`
- `updated_date`

#### Method
- Assign items to price tiers (budget, standard, premium) using category-level quantiles (33rd and 67th percentiles).
- Segment customers by RFM.
- Cross-tabulate: for each (category, price_tier, customer_segment) combination, compute item purchase frequency.
- Identify top-5 items per combo and compute affinity lift.

#### Statistics
- Tier distribution by segment (% of purchases in budget/standard/premium per segment)
- Top-5 item affinity within each (category, tier, segment) combo
- Affinity lift variance across items within combo (measure of item differentiation)
- Segment-tier-item interaction effect (ANOVA with tier and segment as factors)
- Cross-tier rank correlation for items (top items in budget tier vs. same items' rank in premium tier)

#### Visualizations
- Faceted heatmap: categories (rows) × tiers (columns), with segment-colored top items in each cell
- Stacked bar: tier distribution by segment
- Scatter plot: item rank in budget tier (x-axis) vs. premium tier (y-axis), by category color
- Box plot: affinity lift distribution across (category, tier, segment) combos, grouped by tier

#### Decision Rule
- Keep if affinity lift within (category, tier, segment) combos shows >20% variance across items in the same combo, or if cross-tier rank correlation is low (<0.5), indicating tier and segment both meaningfully constrain item choice.

### Idea 15: Brand-to-Brand Switching Patterns

#### Question
When customers switch brands within a category, which brands do they switch to, and are patterns segment-specific?

#### Datasets
- `transaction_full_2025.parquet`
- `items.parquet`

#### Columns to Inspect
- `customer_id`
- `item_id`
- `brand`
- `category_l2`
- `updated_date`
- `quantity`

#### Method
- For each customer-category pair, identify sequential brand purchases (e.g., brand A in period 1 → brand B in period 2).
- Build a brand-to-brand transition matrix per customer segment (rows = from_brand, columns = to_brand, values = count of transitions).
- Compute transition entropy (predictability of switches) and loyalty (diagonal values).
- Normalize rows to get transition probabilities.

#### Statistics
- Brand loyalty rate: proportion of within-brand repeats (high values = customers stick to one brand)
- Switching entropy by segment (higher entropy = less predictable switching)
- Top 3 destination brands for each source brand per segment
- Switching rate by category and segment
- Brand switching correlations with customer RFM (do high-value customers switch more/less?)

#### Visualizations
- Sankey diagram: source brands → destination brands, with flow width = number of transitions, grouped by segment
- Heatmap transition matrix: brands (rows and columns), color = transition probability, separate panels per segment
- Bar chart: switching entropy and loyalty rate by segment
- Scatter plot: brand market share (x-axis) vs. switching-into rate (y-axis) to identify attractive switching targets

#### Decision Rule
- Keep if switching patterns have low entropy (predictable) for at least one major segment, or if certain brands act as clear switching hubs (high in-degree), allowing personalized brand recommendations after purchase.

### Idea 16: Location Preference by Customer Segment

#### Question
Do different customer segments have location preferences, and can location affinity inform local ranking?

#### Datasets
- `transaction_full_2025.parquet`

#### Columns to Inspect
- `customer_id`
- `location_name`
- `updated_date`
- `quantity`

#### Method
- For each customer segment (RFM-based), compute location share: (purchases at location L / total purchases by segment).
- Calculate location concentration: HHI across locations per segment.
- Compute segment-location affinity lift: (segment purchase rate at location) / (global purchase rate at location).

#### Statistics
- Location HHI by segment (values closer to 1 = highly concentrated; closer to 0 = dispersed)
- Top 3 locations per segment with purchase counts and HHI
- Segment-location lift (lift > 1.0 indicates overrepresentation)
- Location concentration variance across segments (higher variance = more segment differentiation)
- Correlation between customer tenure and location concentration

#### Visualizations
- Grouped bar chart: top 10 locations, with segment-colored purchase counts
- Heatmap: locations (rows) × segments (columns), with lift values as color intensity
- Scatter plot: segment size (x-axis) vs. location HHI (y-axis), segment-colored points
- Box plot: location count visited per segment, showing segment variance

#### Decision Rule
- Keep if location HHI by segment is > 0.3 on average (indicating meaningful concentration), or if segment-location lift shows some locations with average lift > 1.5 for a specific segment.

### Idea 17: Item Repeat Propensity by Category

#### Question
Which categories and items have the highest repeat-purchase rates, and does repeat propensity vary significantly by customer segment?

#### Datasets
- `transaction_full_2025.parquet`
- `items.parquet`

#### Columns to Inspect
- `customer_id`
- `item_id`
- `category_l2`
- `updated_date`
- `quantity`

#### Method
- For each item, identify customers who purchased it at least once.
- Measure repeat propensity: (customers who repurchased item / customers who ever bought it).
- Repeat propensity by category: (item repeat propensity averaged within category).
- Segment customers by RFM and compute segment-category repeat correlation.

#### Statistics
- Item-level repeat propensity distribution (mean, median, std across items)
- Category-level repeat propensity (ranked by repeat rate)
- Repeat propensity by customer segment and category (interaction effect)
- Average time-to-repeat for repeat items vs. non-repeat items
- Repeat propensity by item lifecycle stage (new vs. stable vs. declining)
- Correlation between repeat propensity and item price tier

#### Visualizations
- Histogram: item repeat propensity distribution with segment-colored overlays
- Grouped bar chart: top 15 categories ranked by repeat propensity, with segment-colored bars
- Scatter plot: item popularity (x-axis = total purchases) vs. repeat propensity (y-axis), category-colored
- Heatmap: categories (rows) × customer segments (columns), with repeat propensity as values

#### Decision Rule
- Keep if category-level repeat propensity shows >15% variance across categories, or if segment-category interaction is significant (some segments show high repeat in category X while others don't).

### Idea 18: Cross-Category Affinity (Which Categories Co-Purchase)

#### Question
Beyond single-basket associations, do customers have stable cross-category preferences over time?

#### Datasets
- `transaction_full_2025.parquet`
- `items.parquet`

#### Columns to Inspect
- `customer_id`
- `category_l2`
- `updated_date`
- `quantity`

#### Method
- For each customer, split transaction history into 3-month windows.
- Within each window, compute category co-purchase matrix: categories that appear in same customer's purchases.
- Measure pair-wise co-purchase lift: (P(both A and B in window) / (P(A) * P(B))).
- Identify stable pairs: pairs that appear in >50% of windows per customer segment.

#### Statistics
- Category co-purchase lift distribution (overall and by segment)
- Co-purchase stability over time (Jaccard similarity of co-purchase pairs between consecutive periods)
- Top 10 category pairs by average lift within each segment
- Category co-purchase entropy (measure of clustering)
- Hierarchical clustering of categories based on co-purchase distances

#### Visualizations
- Network graph: categories as nodes, co-purchase lift > 1.2 as edges, segment-colored nodes
- Heatmap: category-to-category lift matrix (categories on both axes)
- Sankey or chord diagram: major cross-category flows for top segment
- Dendrogram: hierarchical clustering of categories by co-purchase distance

#### Decision Rule
- Keep if stable category pairs (appearing in >50% of windows) show average lift > 1.5, or if co-purchase clusters are clearly separable (dendrogram shows distinct group distance).

### Idea 19: Item Lifecycle Affinity by Customer Segment

#### Question
Do different customer segments prefer different item lifecycle stages (new vs. established vs. declining)?

#### Datasets
- `transaction_full_2025.parquet`
- `items.parquet`

#### Columns to Inspect
- `customer_id`
- `item_id`
- `updated_date`
- `quantity`
- `category_l2`

#### Method
- Classify items by lifecycle stage using 3-month rolling sales trends: growth (slope > threshold), stable (|slope| ≤ threshold), decline (slope < -threshold).
- For each customer segment, compute purchase rate by item lifecycle stage.
- Calculate stage preference lift: (segment stage purchase rate) / (global stage purchase rate).

#### Statistics
- Stage distribution by segment (% purchases in growth/stable/decline items)
- Stage-segment lift (deviation from global average)
- New-item adoption rate by segment (purchases of items in first 30 days of available data)
- Lifecycle preference variance across segments
- Correlation between customer tenure and lifecycle stage preference (do new customers prefer new items?)

#### Visualizations
- Stacked bar chart: lifecycle stage distribution by segment
- Heatmap: segments (rows) × lifecycle stages (columns), with lift as color
- Scatter plot: customer tenure (x-axis) vs. new-item adoption rate (y-axis), segment-colored
- Line chart: stage purchase rate (y-axis) vs. item age in months (x-axis), segment-colored lines

#### Decision Rule
- Keep if stage preference shows significant variance across segments (e.g., >20% difference between segments in adoption of growth-stage items), or if new-item adoption correlates strongly with customer tenure.

### Idea 20: Manufacturer Affinity Within Category and Tier

#### Question
Within category-tier combinations, which manufacturers are preferred by each customer segment?

#### Datasets
- `transaction_full_2025.parquet`
- `items.parquet`

#### Columns to Inspect
- `customer_id`
- `item_id`
- `manufacturer`
- `category_l2`
- `price`
- `quantity`
- `updated_date`

#### Method
- Assign items to price tiers (budget/standard/premium) within each category using quantiles.
- Segment customers by RFM.
- Cross-tabulate: for each (category, price_tier, customer_segment) combination, compute manufacturer purchase frequency.
- Rank manufacturers per combo and compute affinity lift.

#### Statistics
- Top 3 manufacturers per (category, tier, segment) combo
- Manufacturer concentration (HHI) within each combo
- Manufacturer-segment lift (over-/under-representation by segment)
- Manufacturer market share by category and tier
- Manufacturer consistency across tiers within category

#### Visualizations
- Faceted heatmap: categories (rows) × tiers (columns), with top manufacturers labeled and segment-colored
- Stacked bar chart: top 10 manufacturers by category and segment
- Scatter plot: manufacturer market share in budget tier (x-axis) vs. premium tier (y-axis)
- Grouped bar chart: manufacturer concentration (HHI) by category, segment-colored

#### Decision Rule
- Keep if manufacturer preferences show >15% variance across segments within the same (category, tier) combo, or if certain manufacturers dominate in specific segments (concentration HHI > 0.5 for segment-specific manufacturer sets).

### Idea 21: Customer Location-Category Interaction

#### Question
Do customers at different locations have different category preferences?

#### Datasets
- `transaction_full_2025.parquet`
- `items.parquet`

#### Columns to Inspect
- `customer_id`
- `location_name`
- `category_l2`
- `updated_date`
- `quantity`

#### Method
- For each customer who shops at multiple locations, compute category share distribution at each location.
- Aggregate to location level: (purchases in category / total purchases at location) for all customers at that location.
- Measure category entropy by location (higher entropy = more diverse mix).
- Identify category specialization by location.

#### Statistics
- Location-category affinity lift: (location category share / global category share)
- Category entropy by location
- Top 10 categories per location with share
- Location-category specificity: categories with >1.5x global share at a location
- Category churn across locations: how many categories are in top 10 at all locations vs. location-specific

#### Visualizations
- Heatmap: locations (rows) × categories (columns), with affinity lift as color intensity
- Bar chart: category entropy by location (sorted descending)
- Scatter plot: category entropy (x-axis) vs. location customer count (y-axis)
- Sankey: top 10 categories across locations, showing flow intensity

#### Decision Rule
- Keep if category specialization by location is evident (>20% of categories show lift > 1.5 at specific locations), or if category entropy variance across locations is >0.2.

### Idea 22: Size Tier Affinity by Customer Segment

#### Question
Do customer segments prefer different size buckets (beyond obvious child-age correlation)?

#### Datasets
- `transaction_full_2025.parquet`
- `items.parquet`

#### Columns to Inspect
- `customer_id`
- `item_id`
- `size`
- `updated_date`
- `quantity`

#### Method
- Filter to items with non-null, valid size values (exclude "Không xác định").
- Standardize size labels into ordinal buckets (e.g., XS, S, M, L, XL).
- Segment customers by RFM.
- For each segment, compute size share distribution and size-segment affinity lift.

#### Statistics
- Size distribution by segment (% of purchases in each size category)
- Size-segment lift (over-/under-representation)
- Average size purchased by segment (using ordinal encoding)
- Size concentration (HHI) by segment
- Size stability over customer tenure (early vs. late purchases)

#### Visualizations
- Grouped bar chart: size distribution by segment
- Heatmap: segments (rows) × sizes (columns), with purchase count as color
- Line chart: average size purchased (y-axis) vs. customer purchase number (x-axis), segment-colored
- Scatter plot: customer RFM value (x-axis) vs. average size preference (y-axis)

#### Decision Rule
- Keep if size distribution differs meaningfully across segments (e.g., >20% difference in median size between highest and lowest segments), or if size preferences correlate strongly with customer tenure.

### Idea 23: Item Momentum Within Customer Context

#### Question
Do items with rising sales trends receive preferential purchases from certain customer segments?

#### Datasets
- `transaction_full_2025.parquet`
- `items.parquet`

#### Columns to Inspect
- `customer_id`
- `item_id`
- `updated_date`
- `quantity`
- `category_l2`

#### Method
- Compute 3-month rolling sales trend (slope of quantity sold) for each item.
- Classify items as trending (positive slope), stable, or declining.
- For each customer segment (RFM), measure purchase frequency for trending vs. stable vs. declining items.
- Calculate momentum affinity lift: (segment trending-item purchase rate / global trending-item purchase rate).

#### Statistics
- Trending-item affinity by segment (lift > 1.0 = preference for trending items)
- Trend-segment correlation (do high-RFM customers prefer trending items?)
- Momentum affinity variance across segments
- Repeat rate on trending items vs. stable items by segment
- Time lag between item momentum spike and segment purchase uptake

#### Visualizations
- Grouped bar chart: item momentum affinity by segment
- Scatter plot: item trend slope (x-axis) vs. segment affinity lift (y-axis), segment-colored
- Line chart: cumulative purchases per segment over time, overlay with item momentum indicator
- Heatmap: segments (rows) × momentum bins (columns), showing purchase concentration

#### Decision Rule
- Keep if trending-item affinity shows >15% variance across segments, or if high-RFM segments show lift > 1.3 for trending items, indicating momentum is a usable segment-specific ranking signal.

### Idea 24: Customer-Item-Location Specificity

#### Question
Do customers show location-specific item preferences, or do they maintain consistent preferences across multiple shopping locations?

#### Datasets
- `transaction_full_2025.parquet`
- `items.parquet`

#### Columns to Inspect
- `customer_id`
- `item_id`
- `location_name`
- `updated_date`
- `quantity`

#### Method
- Filter to customers who made purchases at 3+ distinct locations.
- For each customer-location pair, compute item purchase distribution.
- Calculate item overlap: Jaccard similarity of top-20 items across pairs of locations for the same customer.
- Measure location-specific item preference entropy.

#### Statistics
- Jaccard similarity of top-20 items across locations for multi-location customers (1.0 = identical, 0 = no overlap)
- Cross-location consistency (high consistency = same items preferred everywhere)
- Location-specific item entropy (% of items unique to one location per customer)
- Spearman rank correlation of item purchases between locations (measure item rank stability)
- Average item rank shift across locations

#### Visualizations
- Scatter plot: Jaccard similarity (x-axis) vs. customer purchase count (y-axis)
- Heatmap: customer-location pairs (rows) × item overlap percentiles (columns)
- Sankey: top 10 items at location A → overlap at location B, flow width = customer count
- Box plot: Jaccard similarity distribution, grouped by customer RFM segment

#### Decision Rule
- Keep if Jaccard similarity averages <0.6 (indicating meaningful location-specific item preferences), or if rank correlation of items across locations is <0.7, supporting location-aware per-customer ranking.

### Idea 25: Brand-Price-Tier Interaction for Segments

#### Question
Do customer segments show preferences for brand-tier combinations (e.g., premium brands within budget tier)?

#### Datasets
- `transaction_full_2025.parquet`
- `items.parquet`

#### Columns to Inspect
- `customer_id`
- `item_id`
- `brand`
- `price`
- `category_l2`
- `updated_date`
- `quantity`

#### Method
- Assign items to price tiers within each category using quantiles (budget/standard/premium).
- Segment customers by RFM.
- Cross-tabulate: for each (brand, price_tier, customer_segment) combination, compute purchase frequency.
- Identify top brand-tier combos per segment and compute affinity lift.

#### Statistics
- Top 10 brand-tier combos per segment with purchase counts
- Brand-tier-segment affinity lift
- Brand-tier concentration (HHI of brands within each tier by segment)
- Brand prestige index by tier: (premium-tier share / budget-tier share for each brand)
- Segment-specific brand-tier preference variance

#### Visualizations
- Faceted heatmap: brands (rows) × tiers (columns), segments as separate panels, cells colored by purchase count
- Grouped bar chart: top 15 brands, stacked by tier, segment-colored overlays
- Scatter plot: brand market share in budget tier (x-axis) vs. premium tier (y-axis)
- Box plot: prestige index distribution by segment

#### Decision Rule
- Keep if brand-tier combinations show significant segment differences (e.g., >20% variance in top brands per tier across segments), or if prestige index (premium/budget ratio) is segment-specific (correlation with RFM >0.4).

### Idea 26: Category Lifecycle Interaction by Customer Tenure

#### Question
Do customers in different tenure stages (new, established, at-risk) have different lifecycle stage preferences?

#### Datasets
- `transaction_full_2025.parquet`
- `items.parquet`

#### Columns to Inspect
- `customer_id`
- `updated_date`
- `item_id`
- `category_l2`
- `quantity`

#### Method
- Define customer tenure: days since first purchase.
- Classify items by lifecycle stage (growth, stable, decline) using 3-month rolling trends.
- Segment customers by tenure: new (0–90 days), established (91–365 days), veteran (>365 days).
- Cross-tabulate: for each (tenure_segment, item_lifecycle_stage) combination, compute purchase frequency.

#### Statistics
- Item lifecycle stage distribution by tenure segment (% of purchases in growth/stable/decline items)
- Tenure-lifecycle affinity lift
- New-item adoption rate by tenure segment
- Interaction effect: tenure × item_lifecycle on purchase rate (ANOVA)
- Tenure-lifecycle preference correlation

#### Visualizations
- Stacked bar chart: lifecycle stage distribution by tenure segment
- Heatmap: tenure segments (rows) × lifecycle stages (columns), with affinity lift as color
- Line chart: adoption rate of growth-stage items (y-axis) vs. customer tenure (x-axis)
- Scatter plot: customer tenure (x-axis) vs. new-item purchase % (y-axis), with regression line

#### Decision Rule
- Keep if tenure-lifecycle interaction is significant (ANOVA p < 0.05), or if new-item adoption rate differs >20% between new and veteran customers, supporting tenure-aware item lifecycle ranking.

### Idea 27: Hidden Category Preferences

#### Question
Are there sub-category patterns or hidden item clusters that customers consistently buy together but aren't labeled as such?

#### Datasets
- `transaction_full_2025.parquet`
- `items.parquet`

#### Columns to Inspect
- `customer_id`
- `item_id`
- `category_l3`
- `brand`
- `manufacturer`
- `updated_date`
- `quantity`

#### Method
- For each customer segment (RFM), build co-purchase frequency matrix on category_l3 level.
- Apply hierarchical clustering or K-means on co-purchase distances to identify hidden clusters.
- Define clusters as groups of categories that co-purchase together but aren't explicitly linked by label.
- Measure within-cluster cohesion and between-cluster separation.

#### Statistics
- Optimal cluster count by silhouette analysis
- Within-cluster cohesion (average intra-cluster co-purchase lift)
- Between-cluster separation (minimum inter-cluster lift)
- Cluster-segment specificity: proportion of clusters unique to one segment
- Hierarchy depth: number of meaningful cluster levels

#### Visualizations
- Dendrogram: hierarchical clustering of categories by co-purchase distance
- Network graph: categories as nodes, co-purchase edges colored by cluster membership
- Heatmap: co-purchase matrix for top categories with cluster-colored rows/columns
- Scatter plot: cluster size (x-axis) vs. within-cluster cohesion (y-axis), segment-colored

#### Decision Rule
- Keep if >3 distinct, stable clusters emerge with within-cluster cohesion > 1.5 lift, or if clusters are segment-specific (>20% cluster variance across segments).

### Idea 28: Customer Exploration Score by Category

#### Question
Do customers explore (try new items) more aggressively in some categories than others?

#### Datasets
- `transaction_full_2025.parquet`
- `items.parquet`

#### Columns to Inspect
- `customer_id`
- `item_id`
- `category_l2`
- `updated_date`
- `quantity`

#### Method
- For each customer-category pair, compute novelty rate: (first-time items purchased / total items purchased in category).
- Measure exploration entropy: -sum(p_i * log(p_i)) where p_i = proportion of purchases in top-i items.
- Compare exploration rates across categories and customer segments.

#### Statistics
- Exploration entropy by customer and category
- Category exploration rates (mean novelty rate per category)
- Exploration-retention correlation: do customers who explore more in a category also repeat more?
- Exploration bias by customer RFM (do high-value customers explore more?)
- Top-vs-new ratio: (top-item share) / (novel-item share) by category

#### Visualizations
- Grouped bar chart: category exploration entropy ranked by average rate
- Heatmap: categories (rows) × customer RFM segment (columns), with exploration entropy as values
- Scatter plot: exploration rate (x-axis) vs. category repeat rate (y-axis), category-colored
- Box plot: exploration entropy distribution by segment

#### Decision Rule
- Keep if category exploration rates show >25% variance across categories, or if exploration correlates with repeat rate (correlation >0.4), indicating exploration patterns can guide new-item vs. safe-item ranking.

### Idea 29: Competitive Item Substitution

#### Question
Which items act as substitutes within a customer's purchase patterns (rarely bought together)?

#### Datasets
- `transaction_full_2025.parquet`
- `items.parquet`

#### Columns to Inspect
- `customer_id`
- `item_id`
- `category_l2`
- `brand`
- `updated_date`
- `quantity`

#### Method
- Build customer-item purchase matrix (customers × items).
- Compute negative co-purchase frequency: pairs of items rarely purchased together by same customer.
- Define substitution as: items from same category/brand that show negative association (low co-purchase, high mutual exclusivity).
- Compute substitution strength as -1 × co-purchase coefficient (normalized).

#### Statistics
- Substitution strength distribution (mean, median by item pair)
- Top 10 substitution pairs per category
- Substitution clustering: group items into substitution clusters
- Substitution consistency across customer segments
- Substitution-price tier correlation: are substitutes in similar price tiers?

#### Visualizations
- Network graph: items as nodes, substitution edges (negative co-purchase) colored by strength
- Heatmap: top items (rows and columns) with substitution strength as color intensity
- Scatter plot: item co-purchase (x-axis) vs. substitution strength (y-axis)
- Grouped bar chart: top substitution pairs by category

#### Decision Rule
- Keep if >10 strong substitution pairs emerge (substitution strength > 1.5), or if substitution network shows clear clusters, supporting ranking diversification to avoid substitutes.

### Idea 30: Multi-Location Ranking Variance

#### Question
For the same customer, how much does optimal item ranking vary across locations due to assortment differences?

#### Datasets
- `transaction_full_2025.parquet`
- `items.parquet`

#### Columns to Inspect
- `customer_id`
- `item_id`
- `location_name`
- `updated_date`
- `quantity`
- `category_l2`

#### Method
- Filter to customers who made purchases at 3+ distinct locations.
- For each customer-location pair, rank items by purchase frequency (frequency × quantity).
- Compare item ranks across pairs of locations for the same customer.
- Compute rank variance: Spearman correlation of item ranks across locations.

#### Statistics
- Spearman rank correlation of top-20 items across location pairs (1.0 = identical ranking)
- Ranking variance index: 1 - mean(correlation) (higher = more location variance needed)
- Location-specific top items: items in top-20 at one location but not in top-20 at another
- Assortment overlap: Jaccard similarity of available items across customer's shopping locations
- Ranking changes needed per location pair

#### Visualizations
- Scatter plot: Spearman correlation of item ranks (x-axis) vs. assortment overlap (y-axis)
- Heatmap: customer-location pairs (rows) × rank correlation (columns), sorted by variance
- Sankey: top 10 items at location A → ranking at location B, width = rank stability
- Box plot: rank correlation distribution by assortment overlap quartiles

#### Decision Rule
- Keep if mean Spearman correlation across location pairs is <0.6, or if >30% of top-20 items differ between locations, indicating location-aware ranking is necessary for multi-location customers.

## Item-and-Location Structure Bank

These ideas are meant to separate items, categories, brands, manufacturers, sizes, and locations instead of only tracking raw transaction counts. They are the right next layer after the broad EDA pass because they expose which items are structurally different, not just how much gets sold.

### Suggested target

> [!NOTE]
> Columns `brand` and `manufacturer` contain "Không xác định" (Unknown) values. For manufacturers, this is the majority value (~80-90%). Analysis must either handle this as a separate segment or filter it out to avoid aggregate bias.

- These should be treated as the main item-level expansion after the first 30 edge ideas.
- A good cutoff is 8 to 12 item-structure ideas, selected by signal rather than by volume.
- If a location is clearly stocking only a subset of the catalog, that is a useful signal, not a nuisance, and should be analyzed directly.

### Idea 31: Category Trend Curves

#### Question
Do categories have distinct rising, stable, or declining sales trajectories over time, and how stable are these trends?

#### Datasets
- `transaction_full_2025.parquet`
- `items.parquet`

#### Columns to Inspect
- `updated_date`
- `quantity`
- `category_l1`
- `category_l2`
- `category_l3`

#### Method
- Aggregate sales by category_l1, category_l2, and category_l3 over months.
- Fit rolling 3-month trend slopes for each category.
- Compare slopes, volatility, and peak timing across category levels.
- Identify categories with consistent directional movement.

#### Statistics
- Monthly growth rate by category (% month-over-month)
- Rolling 3-month slope distribution per category level
- Trend stability: coefficient of variation of rolling slope
- Volatility (standard deviation of monthly sales)
- Peak month and timing variability across years
- Autocorrelation of category sales at 1, 3, 6-month lags

#### Visualizations
- Line chart: category sales over time, colored by trend direction (rising/stable/declining)
- Scatter plot: category growth rate (x-axis) vs. volatility (y-axis), category-colored
- Faceted line plots: top 10 categories by sales with rolling slope overlay
- Heatmap: categories (rows) × months (columns), colored by growth rate
- Histogram: trend slope distribution by category level

#### Decision Rule
- Keep if some categories show consistent directional movement (slope persistence >0.6 over consecutive 3-month windows) that differs from the overall market (>15% variance across categories).

### Idea 32: Brand Trend Within Category

#### Question
Do brands rise or fall differently inside the same category, and can we forecast brand performance?

#### Datasets
- `transaction_full_2025.parquet`
- `items.parquet`

#### Columns to Inspect
- `updated_date`
- `brand`
- `category_l2`
- `quantity`

#### Method
- For each category_l2, compute brand-level monthly sales shares.
- Track share changes over time for each brand within its category.
- Compare brand share slopes within category to overall brand slopes.
- Identify brands gaining/losing share consistently.

#### Statistics
- Brand share slope within category (positive = gaining share, negative = losing share)
- Share concentration per category (HHI of brand shares)
- Rank correlation of brand shares across consecutive months (stability)
- Brand rank change: Kendall tau correlation across months
- Category-specific brand winners and losers
- Share volatility by brand within category

#### Visualizations
- Stacked area chart: brand sales share over time, categorized by category
- Scatter plot: brand share slope (x-axis) vs. category average slope (y-axis), brand-colored
- Heatmap: brands (rows) × months (columns) within top 5 categories, with share % as color
- Grouped bar chart: top brands by share per category, showing trend direction
- Line chart: brand rank over time within category

#### Decision Rule
- Keep if brand trajectories show stability (>50% of brands maintain consistent trend direction over 6+ months) and differ by category (category × brand interaction significant).

### Idea 33: Manufacturer Stability Within Category

#### Question
Are manufacturers more stable than brands inside category families, and do they provide a useful fallback for ranking?

#### Datasets
- `transaction_full_2025.parquet`
- `items.parquet`

#### Columns to Inspect
- `updated_date`
- `manufacturer`
- `brand`
- `category_l2`
- `quantity`

#### Method
- Repeat the brand-trend logic at manufacturer level.
- Compare within-category concentration (HHI) and share drift between brands and manufacturers.
- Measure manufacturer stability as month-to-month rank correlation.

#### Statistics
- Manufacturer HHI by category vs. brand HHI by category
- Month-over-month manufacturer share volatility vs. brand volatility
- Long-run share change (first 6 months vs. last 6 months)
- Rank stability: Kendall tau of manufacturer ranks across months
- Manufacturer consolidation index: trend in HHI over time
- Manufacturer-brand correlation: do strong brands come from strong manufacturers?

#### Visualizations
- Box plot: share volatility for brands vs. manufacturers, grouped by category
- Scatter plot: manufacturer HHI (x-axis) vs. brand HHI (y-axis), category-colored
- Line chart: HHI over time for brands vs. manufacturers by category
- Heatmap: manufacturers (rows) × months (columns), with share % as color
- Bar chart: rank correlation of manufacturers vs. brands by category

#### Decision Rule
- Keep if manufacturer HHI is >15% more stable than brand HHI (lower volatility), or if manufacturer ranks are more predictable (rank correlation >0.1 higher than brands).

### Idea 34: Size Ladder by Category

#### Question
Do size transitions follow a predictable ladder inside category groups, and can we recommend size upgrades?

#### Datasets
- `transaction_full_2025.parquet`
- `items.parquet`

#### Columns to Inspect
- `size`
- `category_l2`
- `customer_id`
- `updated_date`
- `quantity`

#### Method
- Filter to items with non-null, valid size values.
- Standardize size labels into ordinal buckets (e.g., NB, 3M, 6M, 9M, 12M, 18M for baby items).
- For each category, measure how often adjacent size buckets are bought together or in sequence.
- Build size-transition matrices per category.

#### Statistics
- Size-transition matrix per category (rows = current size, columns = next size)
- Upgrade rate per size: (customers moving up / total repeat customers in size)
- Average time gap between size transitions
- Transition predictability: dominant next-size probability per size
- Category-level size progression pattern (do all categories follow same ladder?)
- Repeat-gap by size bucket (how long customers stay in size before upgrading)

#### Visualizations
- Sankey diagram: size progression flows per category, with width = customer count
- Heatmap: size-transition matrix per category (rows and columns as sizes)
- Box plot: time-to-upgrade distribution by starting size and category
- Scatter plot: customer age-proxy (purchase count) vs. size purchased
- Line chart: average size progression over customer tenure by category

#### Decision Rule
- Keep if size movement is systematic (upgrade rate >40% for repeat customers within size tier) and category-consistent (>60% correlation between categories in transition patterns).

### Idea 35

> [!IMPORTANT]
> Handles "Không xác định" in brand/manufacturer columns.: Price Tier by Category and Brand

#### Question
Do some categories or brands live mostly in one price tier while others span the full range?

#### Datasets
- `transaction_full_2025.parquet`
- `items.parquet`

#### Columns to Inspect
- `price`
- `category_l1`
- `category_l3`
- `brand`
- `item_id`

#### Method
- Assign items to price tiers (budget/standard/premium) using category-level quantiles (33rd, 67th percentile).
- For each category and brand, compute tier distribution.
- Measure tier spread: entropy of tier distribution (higher entropy = wider tier span).

#### Statistics
- Tier entropy by category and brand (0 = single tier, max = equal distribution across tiers)
- Tier concentration ratio: (top tier % / average tier %)
- Median price spread within category/brand
- Category-specific tier ranges (min/max price by tier)
- Brand tier specialization: brands that dominate specific tiers
- Tier-crossing correlation: category tier patterns vs. brand tier patterns

#### Visualizations
- Stacked bar chart: tier distribution by category, brand-colored segments
- Heatmap: categories/brands (rows) × tiers (columns), with % items in each tier
- Box plot: tier entropy distribution by category level
- Scatter plot: category median price (x-axis) vs. tier entropy (y-axis)
- Violin plot: price distribution by tier and category

#### Decision Rule
- Keep if tier structure strongly separates item groups (tier entropy <1.5 for most categories, indicating tier clustering) or if brands show distinct tier specialization (>50% of items in single tier for >20% of brands).

### Idea 36

> [!IMPORTANT]
> Handles "Không xác định" in brand/manufacturer columns.: Location Assortment Coverage

#### Question
Do locations stock the full catalog or only narrow slices of it, and can this explain ranking differences?

#### Datasets
- `transaction_full_2025.parquet`
- `items.parquet`

#### Columns to Inspect
- `location_name`
- `item_id`
- `category_l1`
- `brand`
- `manufacturer`
- `updated_date`

#### Method
- For each location, compute coverage metrics: distinct items, distinct categories, distinct brands, distinct manufacturers.
- Compare coverage rates against global totals.
- Identify locations with specialized vs. comprehensive assortments.
- Measure assortment overlap between location pairs.

#### Statistics
- Coverage rate per location: (items_at_location / total_items_in_catalog)
- Category coverage: (categories_at_location / total_categories)
- Brand coverage: (brands_at_location / total_brands)
- Manufacturer coverage: (manufacturers_at_location / total_manufacturers)
- Assortment HHI by location: concentration of purchases across available items
- Jaccard similarity of assortments between location pairs
- Coverage correlation with location sales volume

#### Visualizations
- Grouped bar chart: coverage metrics by location (top 20 locations by sales)
- Scatter plot: location sales (x-axis) vs. item coverage rate (y-axis)
- Heatmap: locations (rows) × coverage metrics (columns), showing % coverage
- Box plot: assortment overlap (Jaccard) distribution between location pairs
- Scatter plot: number of items stocked (x-axis) vs. sales per item (y-axis)

#### Decision Rule
- Keep if location assortment is uneven enough to explain differences in recommendations (e.g., >30% of locations stock <70% of catalog, or Jaccard similarity <0.6 for many location pairs).

### Idea 37: Location-Specific Category Mix

#### Question
Do locations have stable category preferences, or are they mostly random mixes?

#### Datasets
- `transaction_full_2025.parquet`
- `items.parquet`

#### Columns to Inspect
- `location_name`
- `category_l1`
- `updated_date`
- `quantity`

#### Method
- Compute category shares by location and time windows (e.g., 3-month periods).
- Measure whether some locations are consistently category-heavy.
- Compare location category mix over time for drift.

#### Statistics
- Location-category lift: (location category share / global category share)
- Location mix entropy: category distribution diversity per location
- Month-to-month category drift by location: Jaccard similarity of top-5 categories across periods
- Category specialization by location: categories with lift > 1.3 at specific locations
- Location clustering: locations with similar category mixes
- Stability of location rank within each category over time

#### Visualizations
- Heatmap: locations (rows) × categories (columns), with lift as color intensity
- Stacked bar chart: category distribution by location for top 10 locations
- Line chart: top 3 categories' share over time per location
- Scatter plot: location mix entropy (x-axis) vs. sales volume (y-axis)
- Network graph: locations as nodes, connected if category mix similarity >0.7

#### Decision Rule
- Keep if locations show persistent category specialization (>20% of categories have lift >1.3 at specific locations) or if location category mixes cluster meaningfully (>3 distinct category-mix patterns).

### Idea 38: Location-Item Availability Gap

#### Question
Which items are frequently available in some locations but effectively absent in others?

#### Datasets
- `transaction_full_2025.parquet`
- `items.parquet`

#### Columns to Inspect
- `location_name`
- `item_id`
- `category_l1`
- `updated_date`
- `quantity`

#### Method
- Build an item-location presence matrix from transactions (1 = item sold at location, 0 = no sales).
- For items with global demand, identify locations where they're absent despite local category demand.
- Measure sparsity: proportion of (item, location) pairs with zero sales where category has sales.

#### Statistics
- Item availability rate by location (% of catalog items available)
- Missingness gap: items with high global demand but <3 locations stocking them
- Item-location sparsity: (zero-sale item-location pairs / total category pairs) by category
- Availability correlation: availability at location vs. location sales volume
- Item specialization: items available at few locations despite global popularity
- Availability drift over time: items entering/leaving locations

#### Visualizations
- Heatmap: items (rows, top 50) × locations (columns, top 20), with 1/0 availability
- Scatter plot: item global popularity (x-axis) vs. number of stocking locations (y-axis)
- Box plot: availability rate distribution by location
- Histogram: item availability across locations (how many locations stock each item?)
- Grouped bar: top unavailable items by location category

#### Decision Rule
- Keep if the availability gap is large (>50% of items unavailable at >50% of locations), or if popular items show high location variance (popular items stock at <50% of locations on average).

### Idea 39: Category-Lifecycle Interaction

#### Question
Do lifecycle stages differ meaningfully by category family, requiring category-specific lifecycle tracking?

#### Datasets
- `transaction_full_2025.parquet`
- `items.parquet`

#### Columns to Inspect
- `category_l1`
- `item_id`
- `updated_date`
- `quantity`

#### Method
- Classify items by lifecycle stage within each category_l1 using 3-month rolling trend slopes.
- Compare new/growth/stable/decline proportions across categories.
- Measure category-lifecycle specificity: variance in stage distribution across categories.

#### Statistics
- Stage proportion by category (% of items in each stage within category)
- Category-specific stage prevalence: categories with >40% items in single stage
- Trend slope distribution by category: mean slope and variance
- Category lifecycle curves: aggregate sales trend per category
- Lifecycle stage predictability: logistic regression of stage on category features
- Stage transition rates by category

#### Visualizations
- Stacked bar chart: lifecycle stage distribution by category
- Line chart: aggregate category sales over time with stage annotations
- Heatmap: categories (rows) × lifecycle stages (columns), with proportion as color
- Box plot: item count distribution by lifecycle stage per category
- Scatter plot: category mean trend slope (x-axis) vs. slope volatility (y-axis)

#### Decision Rule
- Keep if lifecycle stage distribution is not uniform across categories (chi-square test p < 0.05), or if >30% of categories have stage-specific characteristics (e.g., categories with high new-item influx vs. low).

### Idea 40: Item Concentration Versus Location Breadth

#### Question
Do highly concentrated items (high demand from few customers) also appear across many locations, or do they stay localized?

#### Datasets
- `transaction_full_2025.parquet`
- `items.parquet`

#### Columns to Inspect
- `item_id`
- `location_name`
- `customer_id`
- `updated_date`
- `quantity`

#### Method
- Compute item-level demand concentration: HHI across customers (items concentrated if HHI > 0.2).
- For each item, count the number of distinct locations where it was sold.
- Analyze correlation between demand concentration and location breadth.
- Profile items: narrow-concentrated, broad-concentrated, narrow-dispersed, broad-dispersed.

#### Statistics
- Correlation between item demand HHI and location count (Pearson r and Spearman rho)
- Item clustering: K-means on (HHI, location_count) to identify item archetypes
- Cluster sizes and characteristics (mean HHI, location breadth per cluster)
- Concentration-breadth interaction: ANOVA or logistic regression
- Concentration by category: which categories have concentrated vs. dispersed items?
- Location breadth distribution: median and variance

#### Visualizations
- Scatter plot: demand concentration HHI (x-axis) vs. location count (y-axis), category-colored
- Scatter plot: overlay cluster assignments (convex hulls or colored regions)
- Heatmap: item archetypes (rows: narrow-conc, broad-conc, etc.) × statistics (columns)
- Box plot: location breadth distribution by concentration quartile
- Histogram: location count distribution for items, with concentration-colored segments

#### Decision Rule
- Keep if concentration-breadth correlation is significant (|r| > 0.3) or if cluster analysis reveals ≥2 distinct item archetypes with different ranking implications (concentrated items might need local boosting, dispersed items might benefit from cross-location recommendations).

### Why this matters

This is the layer that separates real item behavior from aggregate volume. Category trend, size ladders, brand/manufacturer drift, and location assortment gaps are all more likely to produce item-specific recommendation signals than raw transaction totals alone.

## Customer Behavior Analysis Bank

These ideas focus on customer-level patterns and segmentation because the recommendation output is customer→items. Understanding customer lifecycle, category affinity, repeat behavior, and purchase patterns is essential for per-customer ranking and personalization.

### Suggested target

> [!NOTE]
> Columns `brand` and `manufacturer` contain "Không xác định" (Unknown) values. For manufacturers, this is the majority value (~80-90%). Analysis must either handle this as a separate segment or filter it out to avoid aggregate bias.

- These are the highest-priority ideas for a recommendation system because they directly influence ranking for each customer.
- All 10 ideas should be analyzed; filtering is less critical here since customer behavior drives the output structure.
- Handle "Không xác định" values as a separate unknown category; don't drop them—they reveal customer behavior on unlabeled items.

### Idea 41: Customer Lifecycle Stage (New, Active, Dormant, Churned)

#### Question
Can customers be segmented into lifecycle stages based on purchase recency and frequency, and do stages have different purchasing patterns?

#### Datasets
- `transaction_full_2025.parquet`

#### Columns to Inspect
- `customer_id`
- `updated_date`
- `quantity`
- `item_id`

#### Method
- Define customer first-purchase and last-purchase dates.
- Classify customers by last-purchase recency and total purchase count:
  - New: first-purchase within 90 days
  - Active: last purchase within 30 days
  - At-risk: last purchase 31–90 days ago
  - Dormant: last purchase > 90 days ago
  - Churned: no purchase in > 180 days

#### Statistics
- Customer count per stage
- Average quantity per purchase by stage
- Category diversity by stage (entropy)
- Repeat rate by stage
- Average inter-purchase gap by stage

#### Statistics
- Customer count and distribution per stage
- Average quantity per purchase by stage
- Category diversity (entropy) by stage
- Repeat rate by stage (% of stage customers who made 2+ purchases)
- Average inter-purchase gap (days) by stage
- Median purchase value by stage
- Reactivation rate: % of at-risk customers who return

#### Visualizations
- Pie chart: customer distribution across lifecycle stages
- Bar chart: average metrics (frequency, quantity, diversity) by stage
- Violin plot: inter-purchase gap distribution by stage
- Scatter plot: customer purchase count (x-axis) vs. days-since-purchase (y-axis), stage-colored
- Line chart: cumulative reactivation rate over time from at-risk state
- Heatmap: lifecycle stage (rows) × categories (columns), with purchase share as color

#### Decision Rule
- Keep if lifecycle stages show distinct purchasing signatures (e.g., active customers repeat 3+ times more frequently than dormant) and are stable over time (>70% of stage members remain in same stage 6 months later).

### Idea 42: Category Affinity by Customer

#### Question
Do customers concentrate purchases in one or two categories, or do they buy broadly across categories?

#### Datasets
- `transaction_full_2025.parquet`
- `items.parquet`

#### Columns to Inspect
- `customer_id`
- `item_id`
- `category_l1`
- `category_l2`
- `updated_date`
- `quantity`

#### Method
- For each customer, compute category_l1 and category_l1 share distribution: (purchases in category / total purchases).
- Calculate concentration metrics: HHI, entropy, top-1 share, top-3 share.
- Segment customers into concentration quartiles.

#### Statistics
- Customer HHI distribution by category level (mean, median, quartiles)
- Entropy distribution: narrow (entropy < 1.0) vs. broad shoppers (entropy > 2.5)
- Top-1 and top-3 category share statistics
- Correlation between concentration and customer tenure
- Correlation between concentration and total purchase volume
- Correlation between concentration and repeat rate

#### Visualizations
- Histogram: HHI distribution with density curve
- Scatter plot: customer tenure (x-axis) vs. category HHI (y-axis), with regression line
- Scatter plot: total purchases (x-axis) vs. HHI (y-axis), segment-colored
- Boxplot: HHI distribution by lifecycle stage
- Stacked bar chart: category concentration quartiles (x-axis) with top categories (y-axis)
- Heatmap: customers (rows, sampled top 100) × categories (columns), with share as color

#### Decision Rule
- Keep if category affinity is strong and stable (>40% of customers have HHI > 0.3, indicating concentration) and correlates with customer characteristics (correlation with tenure or RFM > 0.3).

### Idea 43

> [!IMPORTANT]
> Handles "Không xác định" in brand/manufacturer columns.: Brand Loyalty Within Preferred Categories

#### Question
Within their preferred categories, do customers stick to a few brands or explore many?

#### Method
- For customers with known category affinity (top 1–2 categories), compute brand concentration.
- Compare brand diversity inside vs. outside preferred categories.

#### Statistics
- Average brand HHI inside preferred categories
- Average brand HHI outside preferred categories
- Difference in brand HHI (inside - outside)
- Repeat-brand rate inside vs. outside preferred categories
- Top-brand share inside preferred categories
- Brand diversity (entropy) comparison
- Correlation between category affinity and brand loyalty

#### Visualizations
- Boxplot: brand HHI inside vs. outside preferred categories
- Scatter plot: brand HHI inside (x-axis) vs. outside (y-axis), with diagonal reference
- Grouped bar chart: repeat-brand rate inside vs. outside
- Violin plot: top-brand share distribution (inside vs. outside)
- Heatmap: customers (rows) × (inside preferred / outside preferred) (columns), with brand HHI as color

#### Decision Rule
- Keep if brand loyalty is significantly higher inside preferred categories (e.g., mean HHI inside > outside by >0.1), indicating category affinity drives brand stickiness.

### Idea 44: Customer-Item Repeat Patterns

#### Question
Which items get repeated purchases, and how does repeat likelihood vary by item type or customer segment?

#### Datasets
- `transaction_full_2025.parquet`
- `items.parquet`

#### Columns to Inspect
- `customer_id`
- `item_id`
- `updated_date`
- `quantity`
- `category_l1`
- `lifecycle_stage` (derived)
- `price_tier` (derived)

#### Method
- Identify repeat purchases: customer_id, item_id pairs appearing >1 time.
- Compute repeat probability per item.
- Compare repeat rates by item lifecycle stage, price tier, category, and customer segment.

#### Statistics
- Repeat probability by item (% of customers who repeat per item)
- Repeat probability distribution by item lifecycle stage
- Repeat probability by price tier
- Repeat probability by customer lifecycle stage
- Average time gap between repeats (days) by category
- Repeat-to-non-repeat ratio by category
- Correlation: repeat likelihood vs. item popularity (purchase count)

#### Visualizations
- Histogram: item repeat probability distribution
- Grouped bar chart: repeat rate by item lifecycle stage
- Heatmap: customer lifecycle stage (rows) × item lifecycle stage (columns), with repeat rate
- Scatter plot: item popularity (total purchases, x-axis) vs. repeat probability (y-axis), category-colored
- Violin plot: time-to-repeat distribution by category
- Box plot: repeat rate by price tier

#### Decision Rule
- Keep if repeat patterns are predictable (e.g., consumables have >30% repeat rate while discretionary items <10%), allowing personalized repeat-item ranking.

### Idea 45: Size Progression and Upgrade Behavior

#### Question
Do customers follow a progression through size buckets (e.g., buying larger sizes over time), and does this pattern vary by customer tenure?

#### Datasets
- `transaction_full_2025.parquet`
- `items.parquet`

#### Columns to Inspect
- `customer_id`
- `item_id`
- `size`
- `updated_date`
- `quantity`
- `category_l1`

#### Method
- Filter to items with valid, non-null size values (exclude "Không xác định").
- Standardize size labels to ordinal scale (e.g., 1=XS, 2=S, 3=M, 4=L, 5=XL).
- For each customer, track size purchases over time.
- Identify customers with multiple purchases and measure size progression (upgrade/downgrade).

#### Statistics
- Size transition matrix by customer (rows = current size, columns = next size)
- Upgrade rate: (customers moving to larger size / customers with repeat purchases)
- Downgrade rate: (customers moving to smaller size / customers with repeat purchases)
- Direction of size drift: average size change over customer lifetime
- Size progression lag: average time between size transitions (days)
- Tenure correlation: newer customers vs. veterans (do new customers buy smaller sizes?)

#### Visualizations
- Sankey diagram: size progression flows, width = customer count
- Heatmap: size-transition matrix (rows and columns = sizes)
- Box plot: time-to-upgrade distribution by starting size
- Scatter plot: customer tenure (x-axis) vs. average size purchased (y-axis)
- Line chart: median size (y-axis) vs. customer purchase sequence number (x-axis)
- Grouped bar chart: upgrade/downgrade rate by category

#### Decision Rule
- Keep if size progression is directional (upgrade rate >40%, downgrade rate <10% for repeat customers) and predictable enough to guide size-aware recommendations.

### Idea 46

> [!IMPORTANT]
> Handles "Không xác định" in brand/manufacturer columns.: Customer Segment Clustering (RFM-style)

#### Question
Can customers be clustered into meaningful purchasing behavior segments based on RFM and behavioral metrics?

#### Datasets
- `transaction_full_2025.parquet`
- `items.parquet`

#### Columns to Inspect
- `customer_id`
- `updated_date`
- `quantity`
- `item_id`
- `category_l1`
- `brand`

#### Method
- For each customer, compute:
  - **Recency**: days since last purchase
  - **Frequency**: total purchase events
  - **Monetary**: total quantity purchased (or alternative: sum of transaction counts)
  - **Diversity**: category entropy or brand entropy
  - **Concentration**: HHI on top category or brand
- Normalize features to [0, 1].
- Apply K-means clustering with optimal K (silhouette analysis or elbow method).
- Assign cluster labels (e.g., "high-value explorers", "loyal budget shoppers").

#### Statistics
- Optimal cluster count (K) by silhouette score
- Cluster sizes and distribution
- Cluster centroids: mean RFM + diversity metrics per cluster
- Within-cluster vs. between-cluster variance ratio
- Silhouette score per cluster
- Segment stability over time: sample two 6-month periods, measure membership persistence (>70% same segment = stable)
- Cluster separation: minimum distance between centroids

#### Visualizations
- Scatter plot: Recency (x-axis) vs. Frequency (y-axis), cluster-colored, size = Monetary
- 3D scatter: Recency, Frequency, Monetary, cluster-colored
- Box plot: diversity and concentration metrics by cluster
- Heatmap: cluster centroids (rows) × RFM metrics (columns), with values normalized
- Bar chart: cluster size distribution
- Silhouette plot: silhouette scores by cluster

#### Decision Rule
- Keep if clusters are stable and distinct (silhouette score >0.4), interpretable (e.g., clear RFM archetypes), and segment membership remains >70% consistent over consecutive 6-month periods.

### Idea 47: Location Affinity by Customer

#### Question
Do customers shop consistently at a small set of locations, or do they vary widely?

#### Datasets
- `transaction_full_2025.parquet`

#### Columns to Inspect
- `customer_id`
- `location_name`
- `updated_date`
- `quantity`

#### Method
- For each customer who made purchases at 2+ locations, compute location_name share distribution.
- Calculate location concentration: HHI across locations.
- Measure top-location share: (purchases at most-frequent location / total purchases).

#### Statistics
- Location HHI distribution by customer (mean, median, quantiles)
- Top-location share distribution (% customers with >50% purchases at one location)
- Correlation between location concentration and customer value (total quantity or purchase frequency)
- Correlation between location concentration and customer tenure
- Location churn: % of customers shopping at multiple locations in period 1 vs. single location in period 2
- Multi-location penetration: % of customers shopping at 3+ locations

#### Visualizations
- Histogram: location HHI distribution
- Scatter plot: customer tenure (x-axis) vs. location HHI (y-axis)
- Box plot: location HHI by lifecycle stage
- Scatter plot: location count (x-axis) vs. customer purchase frequency (y-axis)
- Grouped bar chart: HHI by customer RFM segment
- Heatmap: customers (rows, sampled) × top locations (columns), with share as color

#### Decision Rule
- Keep if location HHI is strong enough to support location-specific ranking (e.g., >50% of customers have HHI > 0.5), or if location affinity correlates with customer value (correlation > 0.3).

### Idea 48: Price Tier Movement by Customer Over Time

#### Question
Do customers shift up or down price tiers over their lifetime, and does this correlate with lifecycle stage?

#### Datasets
- `transaction_full_2025.parquet`
- `items.parquet`

#### Columns to Inspect
- `customer_id`
- `updated_date`
- `price`
- `category_l1`
- `quantity`

#### Method
- Assign each transaction a price tier (budget/standard/premium) based on category-level quantiles (33rd, 67th percentile).
- Split each customer's history into time buckets (e.g., first year, second year, or first 50 purchases, next 50).
- Compute average tier per customer per time bucket.
- Track tier movement: tier in bucket 1 vs. bucket 2.

#### Statistics
- Mean tier by customer tenure decile (customers in first decile of purchases vs. last decile)
- Tier shift direction: (final tier - initial tier) distribution across customers
- Tier movement by lifecycle stage (new vs. active vs. dormant)
- Stability of tier preference: Spearman rank correlation of tier preferences over adjacent time windows
- Premium-to-budget shift rate: (customers moving to higher tier / customers with tier change)
- Tier consistency: % of customers maintaining same tier across time buckets

#### Visualizations
- Scatter plot: initial tier (x-axis) vs. final tier (y-axis), with 45° reference line
- Line chart: mean tier (y-axis) vs. customer tenure decile (x-axis), lifecycle-colored lines
- Box plot: tier shift distribution by customer segment
- Stacked bar chart: tier distribution by tenure phase (early, mid, late purchases)
- Violin plot: tier over time for customers segmented by lifecycle stage

#### Decision Rule
- Keep if tier movement is predictable (>30% of customers show consistent upward or downward tier drift), or if tier movement correlates with lifecycle stage or RFM metrics (correlation > 0.3).

### Idea 49: Purchase Velocity and Seasonality by Customer

#### Question
Do customers have different purchasing speeds and seasonal peaks, and can patterns be personalized?

#### Datasets
- `transaction_full_2025.parquet`
- `items.parquet`

#### Columns to Inspect
- `customer_id`
- `updated_date`
- `quantity`

#### Method
- For each customer, compute monthly purchase counts over their history.
- Calculate customer average purchase frequency per month.
- Measure within-customer seasonality: correlation with calendar month.
- Segment customers by RFM and compute segment-level velocity and seasonality patterns.

#### Statistics
- Customer average purchase frequency per month (mean, median, std)
- Within-customer seasonality: Spearman correlation with calendar month (0 = no seasonality, 1 = strong)
- Seasonal peak month by customer segment (e.g., RFM quartile)
- Seasonal peak month variance: how consistent is seasonal timing across customers?
- Purchase velocity correlation with lifecycle stage
- Purchase velocity correlation with customer RFM metrics

#### Visualizations
- Histogram: customer monthly purchase frequency distribution
- Line chart: mean monthly purchases (y-axis) vs. calendar month (x-axis), segment-colored
- Heatmap: customers (rows, sampled) × calendar months (columns), with purchase count
- Box plot: monthly purchase frequency by customer segment
- Scatter plot: customer frequency (x-axis) vs. seasonality strength (y-axis), segment-colored
- Grouped bar chart: peak purchase month distribution by segment

#### Decision Rule
- Keep if customer velocity patterns are stable (>40% of customers show consistent purchase frequency across months) and seasonal patterns are segment-specific (different segments have different peak months).

### Idea 50: Cross-Category Purchase Patterns by Customer Segment

#### Question
Do different customer segments buy different category combinations, and can this guide bundled recommendations?

#### Datasets
- `transaction_full_2025.parquet`
- `items.parquet`

#### Columns to Inspect
- `customer_id`
- `category_l1`
- `updated_date`
- `quantity`

#### Method
- For each RFM or lifecycle segment, compute category co-purchase rates: (customers buying both category A and B / total customers in segment).
- Identify segment-specific category pairs with high lift.
- Calculate Jaccard similarity of category preferences between segments.

#### Statistics
- Category co-purchase lift by segment: (segment co-purchase rate / global co-purchase rate)
- Top 10 category pairs per segment with co-purchase lift
- Segment-specific bundle support: % of segment customers buying bundle
- Jaccard similarity of category preferences between segments (1.0 = identical, 0 = no overlap)
- Category exclusivity by segment: categories with high segment affinity but low in other segments
- Co-purchase stability: % of top-10 bundles stable across time periods per segment

#### Visualizations
- Heatmap: category pairs (rows and columns) × segments, with co-purchase lift
- Grouped bar chart: top 10 category pairs by segment
- Sankey: major category flows per segment
- Network graph: categories as nodes, segment-colored edges by co-purchase lift
- Scatter plot: global co-purchase rate (x-axis) vs. segment-specific lift (y-axis), segment-colored
- Box plot: co-purchase lift distribution by segment

#### Decision Rule
- Keep if segment-specific bundles show lift > 1.3 and are present in >20% of segment customers, or if Jaccard similarity between segments is <0.7, indicating distinct cross-category patterns.

### Why this matters

Customer analysis is the core of per-customer recommendation output. Lifecycle stage, category affinity, repeat patterns, and segment membership directly drive which items should be ranked highest for each customer. These 10 ideas are the highest-leverage features for a recommendation system.

### Notes on "Không xác định" handling

- Do not drop or filter out "Không xác định" values in size, brand, or manufacturer.
- Treat them as a separate category and analyze customer behavior on unknown items.
- If a customer shows affinity for unknown-sized items or unknown-brand items, that is a signal.
- Compare customers who buy mostly known vs. mostly unknown items.

## Suggested Notebook Order

1. Data discovery and schema validation.
2. Shared Polars transforms.
3. Holiday effect.
4. Brand and manufacturer commitment.
5. Category connections.
6. Size normalization.
7. Seasonality.
8. Replenishment cycles.
9. Basket rules.
10. Customer diversity.
11. Item lifecycle.
12. Price sensitivity and basket consistency.
13. Edge-analysis bank.
14. Item-and-location structure bank.
15. Customer behavior analysis bank.
16. Final ranking summary.

## Final Output Goal

The notebook should end with a short ranking of all ideas by:

- signal strength
- ease of implementation
- potential recommendation value

That ranking should be based on the actual numbers produced by the notebook, not intuition alone.