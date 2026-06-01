import polars as pl
import numpy as np
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')

print("--- Simulating Unsorted Concatenation (Current Behavior) ---")
# 10 users, each has 1 positive and 5 negative candidates
n_users = 10
pos_list, neg_list = [], []

for u in range(n_users):
    pos_list.append({'customer_id': u, 'item_id': 100 + u, 'feat': np.random.rand(), 'target': 1})
    for i in range(5):
        neg_list.append({'customer_id': u, 'item_id': 200 + u * 10 + i, 'feat': np.random.rand(), 'target': 0})

pos_df = pl.DataFrame(pos_list)
neg_df = pl.DataFrame(neg_list)

# Unsorted concat (positives at the top, negatives at the bottom)
df_unsorted = pl.concat([pos_df, neg_df])

# Prepare LGBM data
def prep_lgb(df):
    X = df.select(['feat']).to_numpy()
    y = df['target'].to_numpy()
    g = df.group_by('customer_id', maintain_order=True).len()['len'].to_numpy()
    return X, y, g

X_un, y_un, g_un = prep_lgb(df_unsorted)

print(f"Group sizes: {g_un}")
print(f"y labels for the first few groups: {y_un[:sum(g_un[:3])]}")
print(f"y labels for the last few groups: {y_un[-sum(g_un[-3:]):]}")

# Train tiny lambdarank model on unsorted data
dtrain = lgb.Dataset(X_un, y_un, group=g_un)
params = {'objective': 'lambdarank', 'metric': 'ndcg', 'ndcg_eval_at': [3], 'verbosity': -1}
m_un = lgb.train(params, dtrain, num_boost_round=5)
print(f"Unsorted training NDCG: {m_un.best_score}")

print("\n--- Simulating Sorted Concatenation (Proposed Fix) ---")
df_sorted = df_unsorted.sort('customer_id')
X_sort, y_sort, g_sort = prep_lgb(df_sorted)

print(f"Group sizes: {g_sort}")
print(f"y labels in sorted contiguous groups: {y_sort.tolist()}")

dtrain_sorted = lgb.Dataset(X_sort, y_sort, group=g_sort)
m_sort = lgb.train(params, dtrain_sorted, num_boost_round=5)
print(f"Sorted training NDCG: {m_sort.best_score}")
