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
12. Price sensitivity and basket consistency
13. Final ranking summary.

## Final Output Goal

The notebook should end with a short ranking of all ideas by:

- signal strength
- ease of implementation
- potential recommendation value

That ranking should be based on the actual numbers produced by the notebook, not intuition alone.