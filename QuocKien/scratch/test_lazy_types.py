import polars as pl
import numpy as np
import gc
import pickle
from pathlib import Path
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

# Enable global string cache to guarantee Categorical alignment
pl.enable_string_cache()

T_PATH = '/kaggle/input/datasets/kinonquc/qkindataset2/transaction_full_2025.parquet'
I_PATH = '/kaggle/input/datasets/kinonquc/qkindataset2/items.parquet'

# If running locally, let's use the actual local files
T_PATH = r'd:\CS116\ProjectNumberOne\QuocKien\transaction_full_2025_sample.parquet'
I_PATH = r'd:\CS116\ProjectNumberOne\QuocKien\items_sample.parquet'

# Let's check if the sample parquet files exist. If not, let's find the correct local path.
# Wait, let's look at df_raw loaded path in our previous runs or just load from the real workspace data.
# In Cell 1: T_PATH = '/kaggle/input/datasets/kinonquc/qkindataset2/transaction_full_2025.parquet'
# In our local workspace, where are the sample parquets? Let's check the folder contents of QuocKien or ProjectNumberOne.
