# Simple Guide: Understanding a Two-Stage Recommender System

This guide provides a high-level, conceptual overview of how to build a modern, time-aware recommendation system. It is designed to be easy to understand and can serve as a presentation or discussion guide for your class.

---

## 1. The Core Concept: The "Recommendation Funnel"

In a real-world system, we often have **thousands of items** (e.g., product catalog) and **millions of users**. 
We cannot score every single item for every single user using a complex AI model because it would take too long and consume too much computer memory.

To solve this, we use a **Two-Stage Architecture**:

```
[ All Products in Catalog: 20,000+ Items ]
                 |
                 |  Stage 1: Retrieval (Filter down to a small pool)
                 v
     [ Candidate Pool: ~200 Items ]
                 |
                 |  Stage 2: Ranking (Sort the pool using an AI model)
                 v
   [ Top 10 Recommendations for the User ]
```

---

## 2. Stage 1: Candidate Retrieval (Filtering)
In this stage, we quickly gather a small pool of high-probability products (around 200 items) for each user using simple, fast rules. We pull these candidates from multiple different "channels":

1. **User History (What they bought before):** If a user bought an item in the past, they might want to buy it again (especially repeat staples like milk or diapers).
2. **Replenishment (Predictable repeat cycles):** If a user regularly buys a product every 2 weeks, and it has been 2 weeks since their last purchase, we actively retrieve it as a candidate.
3. **Local Popularity (What sells nearby):** Recommending items that are highly popular at the customer's specific home store location (prevents recommending items that are out of stock locally).
4. **Global Popularity (General Bestsellers):** Recommending top-selling products across all stores (highly useful for new users with no history).
5. **Latent Interests (Collaborative Filtering):** Grouping similar users together. If users who have similar shopping patterns bought an item, we retrieve it for this user too.
6. **Category Anchors:** Finding the user's favorite product category and retrieving the top-selling items in that specific category.

---

## 3. Stage 2: Feature Engineering & AI Ranking
Once we have the pool of ~200 candidates, we create **features** (data points) to help the ranking model understand the relationship between the customer and each candidate item. 

We group these features into three simple levels:

### A. Customer Profile (Who is shopping?)
* How active is this customer? (Total purchases)
* What is their budget bracket? (Average purchase price)
* How loyal are they to specific brands or product categories?

### B. Product Profile (What is the item?)
* Is this item globally popular? (Total units sold)
* Is it sold in many locations, or is it highly local?
* What is its average repeat rate? (Do people buy it again and again?)

### C. Matching Features (Is this item right for this customer?)
* Has this customer bought this exact item before? If so, how long ago?
* Does this item fit the customer's favorite product category or brand?
* **Size Fitting:** Does the clothing/diaper size of this item match the developmental age of the customer's child (inferred from their historical purchases)?
* **Fatigue Protection:** Is the item a one-time purchase (like toys or fashion)? If they already bought it, we penalize it so we don't waste a slot recommending it again.

---

## 4. How the Model Learns to Rank
To train our ranking model, we use a machine learning algorithm called **LightGBM LambdaRank**. 

### The Temporal Train/Test Split
We cannot train the model on the future! To simulate real-world performance, we train the model on historical months and test it strictly on the subsequent month:

* **Train:** Months 1 to 11
* **Test/Evaluate:** Month 12

The model goes through the ~200 candidates for each user and assigns a probability score to each. We sort the list by this score and display the **Top 10** items to the customer.

---

## 5. Summary of Why this Pipeline Works
* **Fast and Scalable:** The two-stage funnel keeps computation lightning-fast.
* **Geographically Smart:** It respects local store inventory and doesn't recommend "ghost items" that are unavailable.
* **Respects Growth:** It tracks diaper and clothing size upgrades over time as children grow.
* **Prevents Fatigue:** It automatically stops recommending non-consumables (like toys) once purchased.
