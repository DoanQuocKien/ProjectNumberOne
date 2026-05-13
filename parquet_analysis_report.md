# Parquet Files Analysis Report

This notebook provides a comprehensive analysis of all Parquet files in the workspace, including:
- Column information and missing values
- Row counts
- Potential relationships and references
- Interesting findings and anomalies
```python
import polars as pl
from pathlib import Path

# Set display options for better readability
pl.Config.set_tbl_rows(100)
pl.Config.set_tbl_cols(-1)
pl.Config.set_fmt_str_lengths(120)

print("Libraries imported successfully!")
```

```text
Libraries imported successfully!
```

## 1. Load and Scan Parquet Files
```python
# Discover all parquet files
parquet_files = []
base_path = Path(r"d:\\CS116\\ProjectNumberOne")
file_sizes_mb = {}

for file in sorted(base_path.rglob("*.parquet")):
    parquet_files.append(file)

print(f"Found {len(parquet_files)} parquet file(s):\n")

# Use Polars lazy scans for fast schema discovery, then collect once for downstream analysis
file_info_rows = []
data_frames = {}
for file in parquet_files:
    lazy_frame = pl.scan_parquet(file)
    schema = lazy_frame.collect_schema()
    df = lazy_frame.collect()
    file_name = file.name
    file_sizes_mb[file_name] = file.stat().st_size / (1024 * 1024)
    data_frames[file_name] = df
    file_info_rows.append({
        'File Path': str(file.relative_to(base_path)),
        'Size (MB)': f"{file_sizes_mb[file_name]:.2f}",
        'Columns': len(schema),
        'Rows': df.height
    })

file_info_df = pl.DataFrame(file_info_rows)
print(file_info_df)
print("\n" + "="*80 + "\n")
```

```text
Found 3 parquet file(s):

shape: (3, 4)
┌───────────────────────────────┬───────────┬─────────┬──────────┐
│ File Path                     ┆ Size (MB) ┆ Columns ┆ Rows     │
│ ---                           ┆ ---       ┆ ---     ┆ ---      │
│ str                           ┆ str       ┆ i64     ┆ i64      │
╞═══════════════════════════════╪═══════════╪═════════╪══════════╡
│ event_full_2025.parquet       ┆ 363.41    ┆ 5       ┆ 37684823 │
│ items.parquet                 ┆ 2.77      ┆ 11      ┆ 29823    │
│ transaction_full_2025.parquet ┆ 632.38    ┆ 8       ┆ 37684823 │
└───────────────────────────────┴───────────┴─────────┴──────────┘

================================================================================

```

## 2. Column Analysis Report

Detailed analysis of columns, data types, meanings, and missing values for each file.
```python
for file_name, df in data_frames.items():
    print(f"\n{'='*80}")
    print(f"FILE: {file_name}")
    print(f"{'='*80}")
    print(f"Shape: {df.height} rows × {df.width} columns\n")
    
    # Create column analysis
    column_analysis = []
    for col in df.columns:
        series = df.get_column(col)
        null_count = series.null_count()
        null_percentage = (null_count / df.height) * 100 if df.height else 0
        dtype = str(series.dtype)
        
        # Infer meaning from column name
        meaning = col.replace('_', ' ').title()
        
        column_analysis.append({
            'Column': col,
            'Type': dtype,
            'NaN Count': null_count,
            'NaN %': f"{null_percentage:.2f}%",
            'Unique Values': series.n_unique(),
            'Meaning': meaning
        })
    
    analysis_df = pl.DataFrame(column_analysis)
    print(analysis_df)
    print()
```

```text

================================================================================
FILE: event_full_2025.parquet
================================================================================
Shape: 37684823 rows × 5 columns

shape: (5, 6)
┌──────────────┬──────────────────────────┬───────────┬───────┬───────────────┬──────────────┐
│ Column       ┆ Type                     ┆ NaN Count ┆ NaN % ┆ Unique Values ┆ Meaning      │
│ ---          ┆ ---                      ┆ ---       ┆ ---   ┆ ---           ┆ ---          │
│ str          ┆ str                      ┆ i64       ┆ str   ┆ i64           ┆ str          │
╞══════════════╪══════════════════════════╪═══════════╪═══════╪═══════════════╪══════════════╡
│ customer_id  ┆ Int32                    ┆ 0         ┆ 0.00% ┆ 2821898       ┆ Customer Id  │
│ item_id      ┆ String                   ┆ 0         ┆ 0.00% ┆ 19891         ┆ Item Id      │
│ quantity     ┆ Int32                    ┆ 0         ┆ 0.00% ┆ 126           ┆ Quantity     │
│ event_type   ┆ String                   ┆ 0         ┆ 0.00% ┆ 1             ┆ Event Type   │
│ updated_date ┆ Datetime(time_unit='us', ┆ 0         ┆ 0.00% ┆ 17326695      ┆ Updated Date │
│              ┆ time_zone=None)          ┆           ┆       ┆               ┆              │
└──────────────┴──────────────────────────┴───────────┴───────┴───────────────┴──────────────┘


================================================================================
FILE: items.parquet
================================================================================
Shape: 29823 rows × 11 columns

shape: (11, 6)
┌──────────────┬────────────────────────────────┬───────────┬───────┬───────────────┬──────────────┐
│ Column       ┆ Type                           ┆ NaN Count ┆ NaN % ┆ Unique Values ┆ Meaning      │
│ ---          ┆ ---                            ┆ ---       ┆ ---   ┆ ---           ┆ ---          │
│ str          ┆ str                            ┆ i64       ┆ str   ┆ i64           ┆ str          │
╞══════════════╪════════════════════════════════╪═══════════╪═══════╪═══════════════╪══════════════╡
│ item_id      ┆ String                         ┆ 0         ┆ 0.00% ┆ 29823         ┆ Item Id      │
│ price        ┆ Decimal(precision=38, scale=4) ┆ 0         ┆ 0.00% ┆ 758           ┆ Price        │
│ category_l1  ┆ String                         ┆ 0         ┆ 0.00% ┆ 15            ┆ Category L1  │
│ category_l2  ┆ String                         ┆ 0         ┆ 0.00% ┆ 136           ┆ Category L2  │
│ category_l3  ┆ String                         ┆ 0         ┆ 0.00% ┆ 479           ┆ Category L3  │
│ category     ┆ String                         ┆ 0         ┆ 0.00% ┆ 1684          ┆ Category     │
│ brand        ┆ String                         ┆ 0         ┆ 0.00% ┆ 990           ┆ Brand        │
│ manufacturer ┆ String                         ┆ 0         ┆ 0.00% ┆ 836           ┆ Manufacturer │
│ description  ┆ String                         ┆ 0         ┆ 0.00% ┆ 9690          ┆ Description  │
│ sale_status  ┆ Int32                          ┆ 0         ┆ 0.00% ┆ 2             ┆ Sale Status  │
│ size         ┆ String                         ┆ 0         ┆ 0.00% ┆ 97            ┆ Size         │
└──────────────┴────────────────────────────────┴───────────┴───────┴───────────────┴──────────────┘


================================================================================
FILE: transaction_full_2025.parquet
================================================================================
Shape: 37684823 rows × 8 columns

shape: (8, 6)
┌───────────────┬──────────────────────────┬───────────┬───────┬───────────────┬───────────────┐
│ Column        ┆ Type                     ┆ NaN Count ┆ NaN % ┆ Unique Values ┆ Meaning       │
│ ---           ┆ ---                      ┆ ---       ┆ ---   ┆ ---           ┆ ---           │
│ str           ┆ str                      ┆ i64       ┆ str   ┆ i64           ┆ str           │
╞═══════════════╪══════════════════════════╪═══════════╪═══════╪═══════════════╪═══════════════╡
│ bill_id       ┆ Int32                    ┆ 0         ┆ 0.00% ┆ 17548336      ┆ Bill Id       │
│ customer_id   ┆ Int32                    ┆ 0         ┆ 0.00% ┆ 2821898       ┆ Customer Id   │
│ item_id       ┆ String                   ┆ 0         ┆ 0.00% ┆ 19891         ┆ Item Id       │
│ price         ┆ Decimal(precision=38,    ┆ 0         ┆ 0.00% ┆ 2427192       ┆ Price         │
│               ┆ scale=4)                 ┆           ┆       ┆               ┆               │
│ quantity      ┆ Int32                    ┆ 0         ┆ 0.00% ┆ 126           ┆ Quantity      │
│ event_type    ┆ String                   ┆ 0         ┆ 0.00% ┆ 1             ┆ Event Type    │
│ updated_date  ┆ Datetime(time_unit='us', ┆ 0         ┆ 0.00% ┆ 17326695      ┆ Updated Date  │
│               ┆ time_zone=None)          ┆           ┆       ┆               ┆               │
│ location_name ┆ String                   ┆ 0         ┆ 0.00% ┆ 987           ┆ Location Name │
└───────────────┴──────────────────────────┴───────────┴───────┴───────────────┴───────────────┘

```

## 3. Row Count Analysis
```python
# Row count analysis
print("ROW COUNT SUMMARY")
print("="*80)
row_count_data = []
for file_name, df in data_frames.items():
    row_count_data.append({
        'File': file_name,
        'Row Count': f"{df.height:,}",
        'Column Count': df.width
    })

row_count_df = pl.DataFrame(row_count_data)
print(row_count_df)
print("\nTotal rows across all files:", sum(df.height for df in data_frames.values()))
print("="*80 + "\n")
```

```text
ROW COUNT SUMMARY
================================================================================
shape: (3, 3)
┌───────────────────────────────┬────────────┬──────────────┐
│ File                          ┆ Row Count  ┆ Column Count │
│ ---                           ┆ ---        ┆ ---          │
│ str                           ┆ str        ┆ i64          │
╞═══════════════════════════════╪════════════╪══════════════╡
│ event_full_2025.parquet       ┆ 37,684,823 ┆ 5            │
│ items.parquet                 ┆ 29,823     ┆ 11           │
│ transaction_full_2025.parquet ┆ 37,684,823 ┆ 8            │
└───────────────────────────────┴────────────┴──────────────┘

Total rows across all files: 75399469
================================================================================

```

## 4. Column Relationships and References

Detect potential foreign key relationships between columns across files.
```python
print("COLUMN RELATIONSHIPS AND REFERENCES")
print("="*80)

# Extract all columns from all files
all_columns_by_file = {}
for file_name, df in data_frames.items():
    all_columns_by_file[file_name] = set(df.columns)

# Look for common column names that might indicate relationships
print("\n1. Columns with matching names across files (potential foreign keys):\n")
relationships_found = False

for file1_name, cols1 in all_columns_by_file.items():
    for file2_name, cols2 in all_columns_by_file.items():
        if file1_name < file2_name:  # Avoid duplicates
            common_cols = cols1 & cols2
            if common_cols:
                relationships_found = True
                print(f"   {file1_name} <---> {file2_name}")
                for col in sorted(common_cols):
                    print(f"      • {col}")

# Look for potential ID columns that might reference other tables
print("\n2. Identified ID/Reference columns:\n")
id_cols = {}
for file_name, df in data_frames.items():
    file_id_cols = [col for col in df.columns if 'id' in col.lower() or 'key' in col.lower()]
    if file_id_cols:
        id_cols[file_name] = file_id_cols
        print(f"   {file_name}:")
        for col in file_id_cols:
            unique_count = df.get_column(col).n_unique()
            print(f"      • {col}: {unique_count} unique values")

print("\n3. Value Overlap Analysis:\n")
# Check for value overlaps between ID columns
for file1_name, file1_df in data_frames.items():
    id_cols_1 = [col for col in file1_df.columns if 'id' in col.lower()]
    for col1 in id_cols_1:
        for file2_name, file2_df in data_frames.items():
            if file1_name != file2_name:
                id_cols_2 = [col for col in file2_df.columns if 'id' in col.lower()]
                for col2 in id_cols_2:
                    # Check for value overlap
                    left_values = set(file1_df.get_column(col1).drop_nulls().to_list())
                    right_values = set(file2_df.get_column(col2).drop_nulls().to_list())
                    overlap = left_values & right_values
                    denominator = max(file1_df.get_column(col1).n_unique(), file2_df.get_column(col2).n_unique())
                    overlap_percentage = (len(overlap) / denominator) * 100 if denominator > 0 else 0
                    
                    if overlap_percentage > 50:  # Significant overlap
                        print(f"   {file1_name}.{col1} <-> {file2_name}.{col2}")
                        print(f"      Overlap: {len(overlap)} values ({overlap_percentage:.1f}%)")

if not relationships_found and not id_cols:
    print("   No obvious relationships detected through naming conventions.")

print("\n" + "="*80 + "\n")
```

```text
COLUMN RELATIONSHIPS AND REFERENCES
================================================================================

1. Columns with matching names across files (potential foreign keys):

   event_full_2025.parquet <---> items.parquet
      • item_id
   event_full_2025.parquet <---> transaction_full_2025.parquet
      • customer_id
      • event_type
      • item_id
      • quantity
      • updated_date
   items.parquet <---> transaction_full_2025.parquet
      • item_id
      • price

2. Identified ID/Reference columns:

   event_full_2025.parquet:
      • customer_id: 2821898 unique values
      • item_id: 19891 unique values
   items.parquet:
      • item_id: 29823 unique values
   transaction_full_2025.parquet:
      • bill_id: 17548336 unique values
      • customer_id: 2821898 unique values
      • item_id: 19891 unique values

3. Value Overlap Analysis:

   event_full_2025.parquet.customer_id <-> transaction_full_2025.parquet.customer_id
      Overlap: 2821898 values (100.0%)
   event_full_2025.parquet.item_id <-> items.parquet.item_id
      Overlap: 19853 values (66.6%)
   event_full_2025.parquet.item_id <-> transaction_full_2025.parquet.item_id
      Overlap: 19891 values (100.0%)
   items.parquet.item_id <-> event_full_2025.parquet.item_id
      Overlap: 19853 values (66.6%)
   items.parquet.item_id <-> transaction_full_2025.parquet.item_id
      Overlap: 19853 values (66.6%)
   transaction_full_2025.parquet.customer_id <-> event_full_2025.parquet.customer_id
      Overlap: 2821898 values (100.0%)
   transaction_full_2025.parquet.item_id <-> event_full_2025.parquet.item_id
      Overlap: 19891 values (100.0%)
   transaction_full_2025.parquet.item_id <-> items.parquet.item_id
      Overlap: 19853 values (66.6%)

================================================================================

```

## 5. Interesting Findings and Anomalies
```python
print("INTERESTING FINDINGS AND ANOMALIES")
print("="*80)

findings = []

# 1. High NaN percentage columns
print("\n1. HIGH MISSING VALUE COLUMNS (> 30% NaN):\n")
high_nan_found = False
for file_name, df in data_frames.items():
    for col in df.columns:
        series = df.get_column(col)
        nan_percentage = (series.null_count() / len(df)) * 100 if len(df) else 0
        if nan_percentage > 30:
            high_nan_found = True
            print(f"   {file_name}.{col}: {nan_percentage:.2f}% missing ({series.null_count()} values)")
            findings.append(f"High missing values in {file_name}.{col}")

if not high_nan_found:
    print("   No columns with >30% missing values found.")

# 2. Duplicate rows analysis
print("\n2. DUPLICATE ROWS ANALYSIS:\n")
for file_name, df in data_frames.items():
    total_duplicates = int(df.is_duplicated().sum())
    if total_duplicates > 0:
        dup_percentage = (total_duplicates / len(df)) * 100 if len(df) else 0
        print(f"   {file_name}: {total_duplicates} duplicate rows ({dup_percentage:.2f}%)")
        findings.append(f"Duplicates found in {file_name}: {total_duplicates} rows")
    else:
        print(f"   {file_name}: No complete duplicate rows")

# 3. Data type anomalies
print("\n3. DATA TYPE ANOMALIES:\n")
for file_name, df in data_frames.items():
    print(f"   {file_name}:")
    for col in df.columns:
        dtype = df.schema[col]
        # Check for objects that might be dates, numbers, etc.
        if dtype == pl.Utf8:
            sample_values = df.get_column(col).drop_nulls().head(3).to_list()
            print(f"      • {col} (string): {sample_values}")

# 4. Column statistics for numeric columns
print("\n4. NUMERIC COLUMNS STATISTICS:\n")
for file_name, df in data_frames.items():
    numeric_cols = [name for name, dtype in df.schema.items() if dtype in pl.NUMERIC_DTYPES]
    if len(numeric_cols) > 0:
        print(f"   {file_name}:")
        for col in numeric_cols:
            series = df.get_column(col)
            min_value = series.min()
            max_value = series.max()
            mean_value = series.mean()
            std_value = series.std()
            print(f"      • {col}")
            print(f"         Range: [{min_value:.2f}, {max_value:.2f}]")
            print(f"         Mean: {mean_value:.2f}, Std: {std_value:.2f}")

# 5. Unique value patterns
print("\n5. UNIQUE VALUE PATTERNS:\n")
for file_name, df in data_frames.items():
    print(f"   {file_name}:")
    for col in df.columns:
        series = df.get_column(col)
        unique_count = series.n_unique()
        total_count = len(df)
        uniqueness_ratio = unique_count / total_count if total_count else 0
        
        # Flag columns with very high or very low uniqueness
        if uniqueness_ratio > 0.95:
            print(f"      • {col}: {unique_count} unique values (HIGHLY UNIQUE - {uniqueness_ratio*100:.1f}%)")
            findings.append(f"Highly unique column: {file_name}.{col}")
        elif uniqueness_ratio < 0.01:
            print(f"      • {col}: {unique_count} unique values (LOW VARIETY - {uniqueness_ratio*100:.2f}%)")
            findings.append(f"Low variety column: {file_name}.{col}")

# 6. Size comparison
print("\n6. FILE SIZE AND DENSITY COMPARISON:\n")
size_data = []
for file_name, df in data_frames.items():
    total_cells = df.height * df.width
    size_data.append({
        'File': file_name,
        'Rows': df.height,
        'Cols': df.width,
        'Total Cells': total_cells,
        'File Size (MB)': f"{file_sizes_mb[file_name]:.2f}",
        'Cells / MB': f"{(total_cells / file_sizes_mb[file_name]):.2f}" if file_sizes_mb[file_name] else 'N/A'
    })

size_df = pl.DataFrame(size_data)
print(size_df)

print("\n" + "="*80)
print("\nSUMMARY OF KEY FINDINGS:")
print("="*80)
if findings:
    for i, finding in enumerate(findings, 1):
        print(f"{i}. {finding}")
else:
    print("No major anomalies detected. Data appears clean and well-structured.")
print("\n")
```

```text
INTERESTING FINDINGS AND ANOMALIES
================================================================================

1. HIGH MISSING VALUE COLUMNS (> 30% NaN):

   No columns with >30% missing values found.

2. DUPLICATE ROWS ANALYSIS:

   event_full_2025.parquet: 3745 duplicate rows (0.01%)
   items.parquet: No complete duplicate rows
   transaction_full_2025.parquet: 360 duplicate rows (0.00%)

3. DATA TYPE ANOMALIES:

   event_full_2025.parquet:
      • item_id (string): ['6767000000002', '2265000000027', '6497000000004']
      • event_type (string): ['Purchase', 'Purchase', 'Purchase']
   items.parquet:
      • item_id (string): ['0008040000046', '0502020000004', '0007010000886']
      • category_l1 (string): ['Đồ chơi & Sách', 'Babycare', 'Babycare']
      • category_l2 (string): ['1Y+', 'Bình sữa, phụ kiện', 'Bình sữa, phụ kiện']
      • category_l3 (string): ['Học tập và phát triển tư duy', 'Núm ty', 'Núm ty']
      • category (string): ['Siêu nhân, robot', 'Núm ty Dr Brown', 'Núm ty Pigeon']
      • brand (string): ['WinWinToys', "Dr.Brown's", 'Pigeon']
      • manufacturer (string): ['Không xác định', 'Không xác định', 'Không xác định']
      • description (string): ['Robo Luồn thun Winwintoys\xa0có hình dạng robot tinh nghịch, bắt mắt, thích hợp cho bé từ 2 tuổi trở lên. Đây là món đồ chơi cực an toàn giúp kích thích khả năng sáng tạo của bé. Đồng thời, sản phẩm còn giúp bé thể hiện sự tỉ mỉ khi vừa chơi vừa học. ĐẶC ĐIỂM NỔI BẬT - Các khớp nối linh hoạt, dễ dàng xoay đổi chiều hướng để tạo nên những mô hình độc đáo tùy theo sự sở thích và khả năng sáng tạo của trẻ - Có màu sắc bắt mắt và nổi bật, phù hợp với sở thích của các bé - Các chi tiết góc cạnh đều được mài nhẵn giúp tránh làm xây xước làn da trẻ nhỏ - Sản phẩm giúp rèn luyện cho bé tính kiên nhẫn, suy nghĩ để vượt khó khăn và phát huy khả năng sáng tạo với nhiều kiểu xếp khác nhau THÔNG TIN SẢN PHẨM - Thương hiệu: Winwintoys - Xuất xứ: Việt Nam', 'Không xác định', 'Với hơn 60 năm qua, trải qua rất nhiều nghiên cứu chuyên sâu về cử động bú mẹ tự nhiên của các em bé trên toàn cầu, các chuyên gia R&D của Pigeon đã phát triển ra\xa0 nhiều dòng núm ti phù hợp với bé. Các dòng núm ti của Pigeon có thiết kế rất đặc biệt với hệ thống van thông khí AVS tránh tình trạng đầy hơi, sặc sữa. - Núm vú vừa khớp với chuyển động của lưỡi vừa hỗ trợ sự phát triển tự nhiên của cơ hàm, răng và cấu trúc xương của bé. Đồng thời cho bé cảm giác như đang bú sữa mẹ, tạo cho các bé cảm giác dễ chịu và thoải mái khi bú bình. - Núm vú được làm từ chất liệu silicone cao cấp, dày gấp 2 lần núm vú thông thường, co giãn và mềm mại giúp bé bú sữa một cách dễ dàng, thoải mái. - Hệ thống van thông khí AVS điều chỉnh áp suất trong bình, điều tiết lượng sữa chảy ra, hạn chế tình trạng đầy hơi, khó tiêu ở trẻ. - Rãnh độc đáo phía trong giúp hạn chế tình trạng sặc sữa của bé (không cần van chống sặc). - Núm vú có thể sử dụng cho tất cả các dòng bình sữa cổ hẹp pigeon HƯỚNG DẪN SỬ DỤNG - Luôn rửa sạch và tiệt trùng núm vú trước và sau khi cho bé sử dụng. - Rửa núm vú bằng dung dịch súc rửa bình sữa và núm vú, đun sôi 3-5 phút hoặc tiệt trùng bằng lò vi sóng, máy tiệt trùng. THÔNG TIN SẢN PHẨM - Tên sản phẩm:\xa0Ty thay bình sữa silicone siêu mềm Pigeon, size S, 2 cái - Xuất xứ:\xa0Indonexia - Chất liệu sản phẩm:\xa0Silicone - Size núm: size S, lỗ ti tròn - Độ tuổi sử dụng: Dành cho bé từ sơ sinh trở lên - Hạn sử dụng: 4 năm kể từ ngày sản xuất - Số lượng : 2 cái / vỉ HƯỚNG DẪN BẢO QUẢN - Bảo quản sản phẩm nơi khô ráo, thoáng mát. -\xa0Chú ý: - Không cho trẻ bú bình khi không có sự giám sát của người lớn. - Kiểm tra kỹ trước và sau khi sử dụng. - Loại bỏ ngay núm vú nếu bị rách, thủng hoặc hư hỏng do trẻ cắn.']
      • size (string): ['Không xác định', 'Không xác định', 'Không xác định']
   transaction_full_2025.parquet:
      • item_id (string): ['6767000000002', '2265000000027', '6497000000004']
      • event_type (string): ['Purchase', 'Purchase', 'Purchase']
      • location_name (string): ['QNG - 255 Quốc Lộ 1A', 'HCM - 88/4 Nguyễn Ảnh Thủ', 'BDU - 272 Đại Lộ Bình Dương']

4. NUMERIC COLUMNS STATISTICS:

```

```text
C:\Users\Quoc Kien\AppData\Local\Temp\ipykernel_31904\1926143121.py:46: DeprecationWarning: `NUMERIC_DTYPES` was deprecated in version 1.0.0. Define your own data type groups or use the `polars.selectors` module for selecting columns of a certain data type.
  numeric_cols = [name for name, dtype in df.schema.items() if dtype in pl.NUMERIC_DTYPES]
```

```text
   event_full_2025.parquet:
      • customer_id
         Range: [15126.00, 9540467.00]
         Mean: 5724827.81, Std: 2609682.28
      • quantity
         Range: [1.00, 1400.00]
         Mean: 1.59, Std: 2.52
   items.parquet:
      • price
         Range: [0.00, 20990000.00]
         Mean: 192117.19, Std: 499585.01
      • sale_status
         Range: [0.00, 1.00]
         Mean: 0.23, Std: 0.42
   transaction_full_2025.parquet:
      • bill_id
         Range: [109123952.00, 158512318.00]
         Mean: 143794772.93, Std: 8346169.21
      • customer_id
         Range: [15126.00, 9540467.00]
         Mean: 5724827.81, Std: 2609682.28
      • price
         Range: [0.00, 20990000.00]
         Mean: 166157.57, Std: 196087.71
      • quantity
         Range: [1.00, 1400.00]
         Mean: 1.59, Std: 2.52

5. UNIQUE VALUE PATTERNS:

   event_full_2025.parquet:
      • item_id: 19891 unique values (LOW VARIETY - 0.05%)
      • quantity: 126 unique values (LOW VARIETY - 0.00%)
      • event_type: 1 unique values (LOW VARIETY - 0.00%)
   items.parquet:
      • item_id: 29823 unique values (HIGHLY UNIQUE - 100.0%)
      • category_l1: 15 unique values (LOW VARIETY - 0.05%)
      • category_l2: 136 unique values (LOW VARIETY - 0.46%)
      • sale_status: 2 unique values (LOW VARIETY - 0.01%)
      • size: 97 unique values (LOW VARIETY - 0.33%)
   transaction_full_2025.parquet:
      • item_id: 19891 unique values (LOW VARIETY - 0.05%)
      • quantity: 126 unique values (LOW VARIETY - 0.00%)
      • event_type: 1 unique values (LOW VARIETY - 0.00%)
      • location_name: 987 unique values (LOW VARIETY - 0.00%)

6. FILE SIZE AND DENSITY COMPARISON:

shape: (3, 6)
┌───────────────────────────────┬──────────┬──────┬─────────────┬────────────────┬────────────┐
│ File                          ┆ Rows     ┆ Cols ┆ Total Cells ┆ File Size (MB) ┆ Cells / MB │
│ ---                           ┆ ---      ┆ ---  ┆ ---         ┆ ---            ┆ ---        │
│ str                           ┆ i64      ┆ i64  ┆ i64         ┆ str            ┆ str        │
╞═══════════════════════════════╪══════════╪══════╪═════════════╪════════════════╪════════════╡
│ event_full_2025.parquet       ┆ 37684823 ┆ 5    ┆ 188424115   ┆ 363.41         ┆ 518490.87  │
│ items.parquet                 ┆ 29823    ┆ 11   ┆ 328053      ┆ 2.77           ┆ 118609.69  │
│ transaction_full_2025.parquet ┆ 37684823 ┆ 8    ┆ 301478584   ┆ 632.38         ┆ 476737.74  │
└───────────────────────────────┴──────────┴──────┴─────────────┴────────────────┴────────────┘

================================================================================

SUMMARY OF KEY FINDINGS:
================================================================================
1. Duplicates found in event_full_2025.parquet: 3745 rows
2. Duplicates found in transaction_full_2025.parquet: 360 rows
3. Low variety column: event_full_2025.parquet.item_id
4. Low variety column: event_full_2025.parquet.quantity
5. Low variety column: event_full_2025.parquet.event_type
6. Highly unique column: items.parquet.item_id
7. Low variety column: items.parquet.category_l1
8. Low variety column: items.parquet.category_l2
9. Low variety column: items.parquet.sale_status
10. Low variety column: items.parquet.size
11. Low variety column: transaction_full_2025.parquet.item_id
12. Low variety column: transaction_full_2025.parquet.quantity
13. Low variety column: transaction_full_2025.parquet.event_type
14. Low variety column: transaction_full_2025.parquet.location_name


```

```python
# List category/brand/manufacturer breakdowns for items.parquet
items_df = data_frames.get("items.parquet")
if items_df is None:
    raise RuntimeError("items.parquet not found in data_frames")

candidate_cols = [
    "category", "category_l1", "category_l2", "category_l3",
    "brand", "manufacturer", "sale_status", "size"
]

available_cols = [c for c in candidate_cols if c in items_df.columns]
if not available_cols:
    print("No target category/brand columns found in items.parquet")
else:
    for col in available_cols:
        print("\n" + "="*80)
        print(f"Column: {col} (value -> number of rows / distinct item_id count)")
        print("-"*80)
        tmp = items_df.with_columns(pl.col(col).fill_null("<NULL>").alias(col))
        breakdown = (
            tmp.group_by(col, maintain_order=False)
               .agg(
                   rows=pl.len(),
                   distinct_item_ids=pl.col("item_id").n_unique()
               )
               .sort("rows", descending=True)
        )
        print(breakdown)
```

```text

================================================================================
Column: category (value -> number of rows / distinct item_id count)
--------------------------------------------------------------------------------
shape: (1_684, 3)
┌────────────────────────────────────┬──────┬───────────────────┐
│ category                           ┆ rows ┆ distinct_item_ids │
│ ---                                ┆ ---  ┆ ---               │
│ str                                ┆ u32  ┆ u32               │
╞════════════════════════════════════╪══════╪═══════════════════╡
│ Bộ bé trai Animo                   ┆ 1108 ┆ 1108              │
│ Bộ bé trai Animo Easy              ┆ 1084 ┆ 1084              │
│ Bộ quần áo bé gái                  ┆ 744  ┆ 744               │
│ Bộ Modal lẻ                        ┆ 740  ┆ 740               │
│ Áo sơ sinh                         ┆ 714  ┆ 714               │
│ Bodysuit                           ┆ 641  ┆ 641               │
│ Bộ bé gái Animo Easy               ┆ 640  ┆ 640               │
│ Quần sơ sinh                       ┆ 620  ┆ 620               │
│ Bộ bé gái Animo                    ┆ 598  ┆ 598               │
│ Áo sơ sinh Animo                   ┆ 567  ┆ 567               │
│ Bộ quần áo bé trai                 ┆ 554  ┆ 554               │
│ Bodysuit cũ                        ┆ 534  ┆ 534               │
│ Giày tập đi khác tồn               ┆ 504  ┆ 504               │
│ Bộ Modal set 2                     ┆ 474  ┆ 474               │
│ Quần, chân váy bé gái              ┆ 459  ┆ 459               │
│ Quần sơ sinh Animo                 ┆ 442  ┆ 442               │
│ Áo bé gái cũ                       ┆ 427  ┆ 427               │
│ Đầm bé gái Animo                   ┆ 411  ┆ 411               │
│ Bodysuit Modal lẻ                  ┆ 349  ┆ 349               │
│ Áo bé gái                          ┆ 330  ┆ 330               │
│ Bodysuit bé trai đùi               ┆ 321  ┆ 321               │
│ Áo bé trai cũ                      ┆ 318  ┆ 318               │
│ Xe mô hình/hoạt hình               ┆ 314  ┆ 314               │
│ Lắp ráp/Xếp hình                   ┆ 311  ┆ 311               │
│ Phụ kiện khác tồn                  ┆ 303  ┆ 303               │
│ Áo bé trai                         ┆ 298  ┆ 298               │
│ Đầm bé gái Animo Easy              ┆ 248  ┆ 248               │
│ Đầm cũ                             ┆ 244  ┆ 244               │
│ Sách tô màu, bóc dán, luyện chữ    ┆ 235  ┆ 235               │
│ Bộ chống muỗi set 2                ┆ 234  ┆ 234               │
│ Giày dép cũ tồn                    ┆ 209  ┆ 209               │
│ Đầm                                ┆ 205  ┆ 205               │
│ Đồ chơi phát nhạc                  ┆ 201  ┆ 201               │
│ Áo khoác                           ┆ 194  ┆ 194               │
│ Bodysuit bé gái đùi                ┆ 180  ┆ 180               │
│ Nhập vai bé gái                    ┆ 171  ┆ 171               │
│ Thú bông                           ┆ 163  ┆ 163               │
│ Bao tay chân                       ┆ 160  ┆ 160               │
│ Bodysuit Modal set 2               ┆ 150  ┆ 150               │
│ Bodysuit đông vải mỏng             ┆ 139  ┆ 139               │
│ Áo bé trai Animo Easy              ┆ 136  ┆ 136               │
│ Nhập vai Unisex                    ┆ 136  ┆ 136               │
│ Sách phát triển kỹ năng, trí tuệ   ┆ 135  ┆ 135               │
│ Vớ                                 ┆ 134  ┆ 134               │
│ Quần bé trai cũ                    ┆ 131  ┆ 131               │
│ Bodysuit bé gái tam giác           ┆ 127  ┆ 127               │
│ Dép sục cho bé tồn                 ┆ 126  ┆ 126               │
│ Bộ đông bé trai dài                ┆ 122  ┆ 122               │
│ Quần bé trai                       ┆ 109  ┆ 109               │
│ Nón tồn cũ                         ┆ 109  ┆ 109               │
│ …                                  ┆ …    ┆ …                 │
│ Height Boosting Step 3             ┆ 1    ┆ 1                 │
│ Fe max                             ┆ 1    ┆ 1                 │
│ Thuyền Xưa                         ┆ 1    ┆ 1                 │
│ Trị muỗi, côn trùng Baby Ganics    ┆ 1    ┆ 1                 │
│ Lactozim                           ┆ 1    ┆ 1                 │
│ Gold Bird                          ┆ 1    ┆ 1                 │
│ Con Bò Cười                        ┆ 1    ┆ 1                 │
│ Metacare Step 2                    ┆ 1    ┆ 1                 │
│ Bubs Goat Step 4                   ┆ 1    ┆ 1                 │
│ Nước rửa bình sữa Kodomo           ┆ 1    ┆ 1                 │
│ Tắm gội thảo dược Dao Care Baby    ┆ 1    ┆ 1                 │
│ Nước rửa tay Babyganics            ┆ 1    ┆ 1                 │
│ Bellamy Mom                        ┆ 1    ┆ 1                 │
│ Lu                                 ┆ 1    ┆ 1                 │
│ Tắm gội thảo dược Cung Đình        ┆ 1    ┆ 1                 │
│ Vinamilk Green Farm                ┆ 1    ┆ 1                 │
│ Kem đánh răng Eucryl               ┆ 1    ┆ 1                 │
│ Kem dưỡng đầu ti Palmer's          ┆ 1    ┆ 1                 │
│ Dung dịch vệ sinh cho bé Saforelle ┆ 1    ┆ 1                 │
│ Bảo nhiên                          ┆ 1    ┆ 1                 │
│ Dược Khoa                          ┆ 1    ┆ 1                 │
│ NAN Organic_Step 1                 ┆ 1    ┆ 1                 │
│ Kem dưỡng ẩm Neutrogena            ┆ 1    ┆ 1                 │
│ Taylor                             ┆ 1    ┆ 1                 │
│ Hạt hỗn hợp                        ┆ 1    ┆ 1                 │
│ Enfa C-sec Step 4                  ┆ 1    ┆ 1                 │
│ Bubs Organic step 2                ┆ 1    ┆ 1                 │
│ Pedia Kenji Step 4                 ┆ 1    ┆ 1                 │
│ Kingphar                           ┆ 1    ┆ 1                 │
│ Dầu massage Cetaphil               ┆ 1    ┆ 1                 │
│ Xịt mũi Navax                      ┆ 1    ┆ 1                 │
│ Dầu Hạt cải                        ┆ 1    ┆ 1                 │
│ Aptamil Úc Step 1                  ┆ 1    ┆ 1                 │
│ Anlene                             ┆ 1    ┆ 1                 │
│ Phấn rôm & phấn thơm Pigeon        ┆ 1    ┆ 1                 │
│ Kem chống hăm Sudocream            ┆ 1    ┆ 1                 │
│ Khẩu trang cho bé PIGEON           ┆ 1    ┆ 1                 │
│ Mason                              ┆ 1    ┆ 1                 │
│ Ildong                             ┆ 1    ┆ 1                 │
│ Varna Colostrum Adult              ┆ 1    ┆ 1                 │
│ Abbott Grow Step 1                 ┆ 1    ┆ 1                 │
│ Immunel Step 3                     ┆ 1    ┆ 1                 │
│ Phụ kiện Bebear                    ┆ 1    ┆ 1                 │
│ Nồi nấu chậm ngừng bán             ┆ 1    ┆ 1                 │
│ Tẩy tế bào chết cơ thể Cocoon      ┆ 1    ┆ 1                 │
│ Aptamil Úc Step 4                  ┆ 1    ┆ 1                 │
│ Tanabiki                           ┆ 1    ┆ 1                 │
│ Alphagen Step 4                    ┆ 1    ┆ 1                 │
│ Dung dịch vệ sinh phụ nữ Fremfesh  ┆ 1    ┆ 1                 │
│ Tẩy tế bào chết da mặt Cocoon      ┆ 1    ┆ 1                 │
└────────────────────────────────────┴──────┴───────────────────┘

================================================================================
Column: category_l1 (value -> number of rows / distinct item_id count)
--------------------------------------------------------------------------------
shape: (15, 3)
┌────────────────────────┬───────┬───────────────────┐
│ category_l1            ┆ rows  ┆ distinct_item_ids │
│ ---                    ┆ ---   ┆ ---               │
│ str                    ┆ u32   ┆ u32               │
╞════════════════════════╪═══════╪═══════════════════╡
│ Thời trang             ┆ 16703 ┆ 16703             │
│ Phụ kiện               ┆ 3516  ┆ 3516              │
│ Đồ chơi & Sách         ┆ 3116  ┆ 3116              │
│ Babycare               ┆ 2054  ┆ 2054              │
│ Thực phẩm cho bé       ┆ 1051  ┆ 1051              │
│ Textile                ┆ 572   ┆ 572               │
│ Thực phẩm cho gia đình ┆ 487   ┆ 487               │
│ Tã                     ┆ 471   ┆ 471               │
│ Sữa                    ┆ 469   ┆ 469               │
│ Hóa mỹ phẩm gia đình   ┆ 392   ┆ 392               │
│ Hóa mỹ phẩm cho bé     ┆ 355   ┆ 355               │
│ TPCN                   ┆ 281   ┆ 281               │
│ Sữa nước               ┆ 195   ┆ 195               │
│ Vệ sinh                ┆ 151   ┆ 151               │
│ Gói Hội Viên           ┆ 10    ┆ 10                │
└────────────────────────┴───────┴───────────────────┘

================================================================================
Column: category_l2 (value -> number of rows / distinct item_id count)
--------------------------------------------------------------------------------
shape: (136, 3)
┌────────────────────────────┬──────┬───────────────────┐
│ category_l2                ┆ rows ┆ distinct_item_ids │
│ ---                        ┆ ---  ┆ ---               │
│ str                        ┆ u32  ┆ u32               │
╞════════════════════════════╪══════╪═══════════════════╡
│ Cơ cấu hàng cũ             ┆ 9138 ┆ 9138              │
│ Thời trang bé trai         ┆ 2861 ┆ 2861              │
│ Thời trang bé gái          ┆ 2432 ┆ 2432              │
│ Modal kháng khuẩn          ┆ 2025 ┆ 2025              │
│ 1Y+                        ┆ 1685 ┆ 1685              │
│ Quần áo & Phụ kiện sơ sinh ┆ 1558 ┆ 1558              │
│ Sách & VPP                 ┆ 740  ┆ 740               │
│ 0-1Y                       ┆ 691  ┆ 691               │
│ Giày tập đi                ┆ 671  ┆ 671               │
│ Bình sữa, phụ kiện         ┆ 535  ┆ 535               │
│ Giày dép 1-3Y              ┆ 485  ┆ 485               │
│ Thời trang đông            ┆ 452  ┆ 452               │
│ Nón                        ┆ 433  ┆ 433               │
│ Đồ dùng ăn uống            ┆ 405  ┆ 405               │
│ Chăm sóc gia đình          ┆ 321  ┆ 321               │
│ Đồ dùng vệ sinh            ┆ 254  ┆ 254               │
│ Bánh & Kẹo cho bé          ┆ 221  ┆ 221               │
│ TPCN cho bé                ┆ 206  ┆ 206               │
│ Đồ dùng ra ngoài           ┆ 202  ┆ 202               │
│ Snack ăn dặm               ┆ 197  ┆ 197               │
│ Đồ uống                    ┆ 172  ┆ 172               │
│ Mì & Đồ khô ăn liền        ┆ 170  ┆ 170               │
│ Thiết bị điện gia dụng     ┆ 149  ┆ 149               │
│ Phụ kiện khác              ┆ 146  ┆ 146               │
│ TP từ sữa (bảo quản lạnh)  ┆ 144  ┆ 144               │
│ Vệ sinh cho bé             ┆ 143  ┆ 143               │
│ Bé ngủ                     ┆ 124  ┆ 124               │
│ Cơ cấu hàng tồn            ┆ 119  ┆ 119               │
│ Dầu ăn & Gia vị            ┆ 114  ┆ 114               │
│ Khăn gia đình              ┆ 100  ┆ 100               │
│ Gối                        ┆ 100  ┆ 100               │
│ Chăm sóc cơ thể            ┆ 87   ┆ 87                │
│ Huggies                    ┆ 83   ┆ 83                │
│ Khăn em bé                 ┆ 83   ┆ 83                │
│ Chăn                       ┆ 76   ┆ 76                │
│ Bột ăn dặm                 ┆ 75   ┆ 75                │
│ Sữa bột pha sẵn            ┆ 73   ┆ 73                │
│ Chăm sóc tóc               ┆ 73   ┆ 73                │
│ Merries                    ┆ 72   ┆ 72                │
│ Bobby                      ┆ 71   ┆ 71                │
│ Chăn ga gối gia đình       ┆ 69   ┆ 69                │
│ Chăm sóc sức khỏe bé       ┆ 67   ┆ 67                │
│ Nestle                     ┆ 66   ┆ 66                │
│ Thức ăn nghiền cho bé      ┆ 65   ┆ 65                │
│ Chăm sóc da                ┆ 65   ┆ 65                │
│ Đồ dùng cho mẹ             ┆ 64   ┆ 64                │
│ Thực phẩm đông lạnh        ┆ 63   ┆ 63                │
│ Kẹo                        ┆ 60   ┆ 60                │
│ Enfa                       ┆ 60   ┆ 60                │
│ Cháo ăn dặm                ┆ 58   ┆ 58                │
│ …                          ┆ …    ┆ …                 │
│ Thơm không gian            ┆ 16   ┆ 16                │
│ Bubs                       ┆ 15   ┆ 15                │
│ Khăn khô                   ┆ 14   ┆ 14                │
│ Genki                      ┆ 13   ┆ 13                │
│ Sữa chua uống              ┆ 13   ┆ 13                │
│ Nutricare                  ┆ 11   ┆ 11                │
│ Wakodo                     ┆ 11   ┆ 11                │
│ Elprairie                  ┆ 10   ┆ 10                │
│ Sữa hạt                    ┆ 10   ┆ 10                │
│ Trái cây                   ┆ 10   ┆ 10                │
│ Glico                      ┆ 10   ┆ 10                │
│ Aptamil                    ┆ 10   ┆ 10                │
│ Caryn                      ┆ 9    ┆ 9                 │
│ Whito                      ┆ 9    ┆ 9                 │
│ A2 milk company            ┆ 9    ┆ 9                 │
│ Hipp                       ┆ 9    ┆ 9                 │
│ Meiji                      ┆ 8    ┆ 8                 │
│ Animo                      ┆ 8    ┆ 8                 │
│ Sức khỏe gia đình          ┆ 7    ┆ 7                 │
│ Gói Hội Viên Mass          ┆ 7    ┆ 7                 │
│ Whoopee                    ┆ 7    ┆ 7                 │
│ Blackmores                 ┆ 6    ┆ 6                 │
│ Ildong                     ┆ 5    ┆ 5                 │
│ Mứt & Bơ phết              ┆ 5    ┆ 5                 │
│ Baby love                  ┆ 5    ┆ 5                 │
│ Humana                     ┆ 5    ┆ 5                 │
│ Bellamy                    ┆ 5    ┆ 5                 │
│ Confidence                 ┆ 5    ┆ 5                 │
│ TP bảo quản lạnh           ┆ 5    ┆ 5                 │
│ Kabrita                    ┆ 4    ┆ 4                 │
│ Snow Brand                 ┆ 4    ┆ 4                 │
│ Sweety                     ┆ 4    ┆ 4                 │
│ Alphagen                   ┆ 4    ┆ 4                 │
│ Bông tẩy trang             ┆ 4    ┆ 4                 │
│ Meiji Nội địa              ┆ 4    ┆ 4                 │
│ Aptamil Úc                 ┆ 4    ┆ 4                 │
│ Purelac                    ┆ 3    ┆ 3                 │
│ Gói Hội Viên BẦU           ┆ 3    ┆ 3                 │
│ Bột lắc sữa                ┆ 3    ┆ 3                 │
│ Tã mẫu thử                 ┆ 3    ┆ 3                 │
│ Anmum                      ┆ 2    ┆ 2                 │
│ Kid Essential              ┆ 2    ┆ 2                 │
│ XO                         ┆ 2    ┆ 2                 │
│ Maeil                      ┆ 2    ┆ 2                 │
│ Sữa cho mẹ                 ┆ 1    ┆ 1                 │
│ Appekidz                   ┆ 1    ┆ 1                 │
│ Bông gạc                   ┆ 1    ┆ 1                 │
│ SP mùa vụ                  ┆ 1    ┆ 1                 │
│ Dad and Me                 ┆ 1    ┆ 1                 │
│ Anlene                     ┆ 1    ┆ 1                 │
└────────────────────────────┴──────┴───────────────────┘

================================================================================
Column: category_l3 (value -> number of rows / distinct item_id count)
--------------------------------------------------------------------------------
shape: (479, 3)
┌────────────────────────────────┬──────┬───────────────────┐
│ category_l3                    ┆ rows ┆ distinct_item_ids │
│ ---                            ┆ ---  ┆ ---               │
│ str                            ┆ u32  ┆ u32               │
╞════════════════════════════════╪══════╪═══════════════════╡
│ Thời trang bé trai, bé gái cũ  ┆ 5188 ┆ 5188              │
│ Bộ bé trai                     ┆ 2192 ┆ 2192              │
│ Quần, áo & phụ kiện sơ sinh cũ ┆ 1768 ┆ 1768              │
│ Bộ Modal                       ┆ 1458 ┆ 1458              │
│ Bộ bé gái                      ┆ 1238 ┆ 1238              │
│ Giày dép tồn                   ┆ 1093 ┆ 1093              │
│ Đầm bé gái                     ┆ 659  ┆ 659               │
│ Sách                           ┆ 610  ┆ 610               │
│ Áo                             ┆ 608  ┆ 608               │
│ Bodysuit Modal                 ┆ 567  ┆ 567               │
│ Học tập và phát triển tư duy   ┆ 533  ┆ 533               │
│ Quần                           ┆ 461  ┆ 461               │
│ Phụ  kiện tồn                  ┆ 458  ┆ 458               │
│ Bodysuit bé trai               ┆ 433  ┆ 433               │
│ Nhập vai                       ┆ 371  ┆ 371               │
│ Phương tiện giao thông         ┆ 370  ┆ 370               │
│ Bodysuit bé gái                ┆ 351  ┆ 351               │
│ Giày tập đi 149k               ┆ 289  ┆ 289               │
│ Phát triển giác quan           ┆ 267  ┆ 267               │
│ Dép sục                        ┆ 265  ┆ 265               │
│ Phụ kiện khác                  ┆ 259  ┆ 259               │
│ Dụng cụ ăn uống                ┆ 220  ┆ 220               │
│ Giày chút chít 179k            ┆ 208  ┆ 208               │
│ Phụ kiện cho bé                ┆ 205  ┆ 205               │
│ Đồ dùng gia đình               ┆ 203  ┆ 203               │
│ Đồ chơi nhồi bông              ┆ 197  ┆ 197               │
│ Bộ đông                        ┆ 191  ┆ 191               │
│ Bodysuit đông                  ┆ 191  ┆ 191               │
│ Áo bé trai                     ┆ 190  ┆ 190               │
│ Nón sơ sinh                    ┆ 179  ┆ 179               │
│ Bình sữa                       ┆ 178  ┆ 178               │
│ Bao tay chân, nón              ┆ 169  ┆ 169               │
│ Sandal bé trai                 ┆ 160  ┆ 160               │
│ Bình sữa, phụ kiện ngừng bán   ┆ 154  ┆ 154               │
│ Đồ bầu                         ┆ 142  ┆ 142               │
│ Nón bé trai                    ┆ 141  ┆ 141               │
│ Vận động ngoài trời            ┆ 138  ┆ 138               │
│ Đồ chơi nước                   ┆ 137  ┆ 137               │
│ Phụ kiện làm đẹp tồn           ┆ 121  ┆ 121               │
│ Bình tập uống                  ┆ 120  ┆ 120               │
│ Áo bé gái                      ┆ 118  ┆ 118               │
│ Bodysuit                       ┆ 115  ┆ 115               │
│ Nón bé gái                     ┆ 113  ┆ 113               │
│ Tắm gội cho bé                 ┆ 111  ┆ 111               │
│ Nón tồn                        ┆ 109  ┆ 109               │
│ Xúc xắc                        ┆ 109  ┆ 109               │
│ Cài, kẹp, cột                  ┆ 108  ┆ 108               │
│ Kẹo đồ chơi                    ┆ 101  ┆ 101               │
│ Núm ty                         ┆ 100  ┆ 100               │
│ Thế Giới Động Vật              ┆ 95   ┆ 95                │
│ …                              ┆ …    ┆ …                 │
│ Dessert                        ┆ 2    ┆ 2                 │
│ Vitamin E                      ┆ 1    ┆ 1                 │
│ Mắc ca                         ┆ 1    ┆ 1                 │
│ Lecithin                       ┆ 1    ┆ 1                 │
│ Xịt côn trùng                  ┆ 1    ┆ 1                 │
│ Australia's Own                ┆ 1    ┆ 1                 │
│ Dầu gió                        ┆ 1    ┆ 1                 │
│ Anlene                         ┆ 1    ┆ 1                 │
│ Dad and Me                     ┆ 1    ┆ 1                 │
│ Kit test SARS-CoV-2            ┆ 1    ┆ 1                 │
│ Bubs Full Cream                ┆ 1    ┆ 1                 │
│ Oggi Gold                      ┆ 1    ┆ 1                 │
│ Nước rửa đa năng               ┆ 1    ┆ 1                 │
│ Hỗ trợ trí não                 ┆ 1    ┆ 1                 │
│ Appekidz                       ┆ 1    ┆ 1                 │
│ Máy tạo ẩm                     ┆ 1    ┆ 1                 │
│ Tẩy tế bào chết cơ thể         ┆ 1    ┆ 1                 │
│ Similac Total Comfort          ┆ 1    ┆ 1                 │
│ Nutramigen                     ┆ 1    ┆ 1                 │
│ Phô mai que                    ┆ 1    ┆ 1                 │
│ Dầu màng tang                  ┆ 1    ┆ 1                 │
│ Máy rửa bình sữa               ┆ 1    ┆ 1                 │
│ Nước uống có ga                ┆ 1    ┆ 1                 │
│ Grow Plus trắng                ┆ 1    ┆ 1                 │
│ Dung dịch vệ sinh cho bé       ┆ 1    ┆ 1                 │
│ Mật ong                        ┆ 1    ┆ 1                 │
│ Óc chó                         ┆ 1    ┆ 1                 │
│ Bàn ủi hơi nước                ┆ 1    ┆ 1                 │
│ Thịt viên                      ┆ 1    ┆ 1                 │
│ Similac Neosure                ┆ 1    ┆ 1                 │
│ PreNAN                         ┆ 1    ┆ 1                 │
│ A2 milk company                ┆ 1    ┆ 1                 │
│ Ngủ ngon                       ┆ 1    ┆ 1                 │
│ Máy vắt cam                    ┆ 1    ┆ 1                 │
│ Sữa ong chúa                   ┆ 1    ┆ 1                 │
│ Colosbaby Pedia                ┆ 1    ┆ 1                 │
│ Khăn bếp                       ┆ 1    ┆ 1                 │
│ A2 Immune                      ┆ 1    ┆ 1                 │
│ Boost                          ┆ 1    ┆ 1                 │
│ Máy hút bụi                    ┆ 1    ┆ 1                 │
│ Bông tẩy trang Thổ Nhĩ Kì      ┆ 1    ┆ 1                 │
│ Men tiêu hóa                   ┆ 1    ┆ 1                 │
│ Bánh Trung Thu                 ┆ 1    ┆ 1                 │
│ Tẩy tế bào chết da mặt         ┆ 1    ┆ 1                 │
│ Bông gạc                       ┆ 1    ┆ 1                 │
│ NutriBoost                     ┆ 1    ┆ 1                 │
│ Dừa                            ┆ 1    ┆ 1                 │
│ Lactoferrin                    ┆ 1    ┆ 1                 │
│ Thanh cua                      ┆ 1    ┆ 1                 │
│ Enfa gentle care               ┆ 1    ┆ 1                 │
└────────────────────────────────┴──────┴───────────────────┘

================================================================================
Column: brand (value -> number of rows / distinct item_id count)
--------------------------------------------------------------------------------
shape: (990, 3)
┌──────────────────────┬──────┬───────────────────┐
│ brand                ┆ rows ┆ distinct_item_ids │
│ ---                  ┆ ---  ┆ ---               │
│ str                  ┆ u32  ┆ u32               │
╞══════════════════════╪══════╪═══════════════════╡
│ Animo                ┆ 9260 ┆ 9260              │
│ Không xác định       ┆ 6526 ┆ 6526              │
│ CF (ConCung Fashion) ┆ 5443 ┆ 5443              │
│ Thương hiệu khác     ┆ 712  ┆ 712               │
│ Con Cưng             ┆ 492  ┆ 492               │
│ TOYCITY              ┆ 305  ┆ 305               │
│ ConCung Good         ┆ 211  ┆ 211               │
│ Nous                 ┆ 140  ┆ 140               │
│ Pigeon               ┆ 120  ┆ 120               │
│ Mesuca               ┆ 114  ┆ 114               │
│ KUKU                 ┆ 102  ┆ 102               │
│ Laluna               ┆ 77   ┆ 77                │
│ Konbini              ┆ 63   ┆ 63                │
│ Bobby                ┆ 61   ┆ 61                │
│ Lock&Lock (Hàn Quốc) ┆ 59   ┆ 59                │
│ CYPRESS TOYS         ┆ 58   ┆ 58                │
│ Joie                 ┆ 58   ┆ 58                │
│ Đinh Tị              ┆ 56   ┆ 56                │
│ Heinz                ┆ 54   ┆ 54                │
│ NS Minh Long         ┆ 51   ┆ 51                │
│ Edison (Hàn Quốc)    ┆ 50   ┆ 50                │
│ MAM                  ┆ 48   ┆ 48                │
│ Ivenet               ┆ 46   ┆ 46                │
│ Inochi               ┆ 44   ┆ 44                │
│ PAPA                 ┆ 44   ┆ 44                │
│ Polesie Toys         ┆ 42   ┆ 42                │
│ Philips Avent        ┆ 42   ┆ 42                │
│ PUKU                 ┆ 41   ┆ 41                │
│ Mollis               ┆ 41   ┆ 41                │
│ Nhã Nam              ┆ 40   ┆ 40                │
│ Wesser               ┆ 40   ┆ 40                │
│ NIN House            ┆ 40   ┆ 40                │
│ Nhựa Tân Phú         ┆ 40   ┆ 40                │
│ Carrot               ┆ 39   ┆ 39                │
│ Merries Nhật         ┆ 39   ┆ 39                │
│ Tuyết Mai            ┆ 38   ┆ 38                │
│ Gluck                ┆ 37   ┆ 37                │
│ Marcus & Marcus      ┆ 37   ┆ 37                │
│ Autoru               ┆ 36   ┆ 36                │
│ Kim Đồng             ┆ 35   ┆ 35                │
│ HiPP                 ┆ 34   ┆ 34                │
│ Quà tặng không bán   ┆ 33   ┆ 33                │
│ Moony Blue           ┆ 33   ┆ 33                │
│ Long Thủy            ┆ 32   ┆ 32                │
│ BeBéar (Bebamour)    ┆ 32   ┆ 32                │
│ Tân Việt             ┆ 31   ┆ 31                │
│ Sài Gòn Food         ┆ 31   ┆ 31                │
│ Huggies Dry          ┆ 31   ┆ 31                │
│ Tommee Tippee        ┆ 31   ┆ 31                │
│ Molfix               ┆ 31   ┆ 31                │
│ …                    ┆ …    ┆ …                 │
│ Clean                ┆ 1    ┆ 1                 │
│ Olitsea              ┆ 1    ┆ 1                 │
│ Daesang              ┆ 1    ┆ 1                 │
│ Sunmum               ┆ 1    ┆ 1                 │
│ Saforelle Bebe       ┆ 1    ┆ 1                 │
│ Cellox               ┆ 1    ┆ 1                 │
│ Wellkid              ┆ 1    ┆ 1                 │
│ Anlene               ┆ 1    ┆ 1                 │
│ ValueMED Pharma      ┆ 1    ┆ 1                 │
│ KINOHIMITSU          ┆ 1    ┆ 1                 │
│ Poli                 ┆ 1    ┆ 1                 │
│ MAXI COSI            ┆ 1    ┆ 1                 │
│ Harrys Brioche       ┆ 1    ┆ 1                 │
│ Bestolio             ┆ 1    ┆ 1                 │
│ Elle & Vire          ┆ 1    ┆ 1                 │
│ Welkids              ┆ 1    ┆ 1                 │
│ Vương Tràm Hương     ┆ 1    ┆ 1                 │
│ Arm & Hammer         ┆ 1    ┆ 1                 │
│ HYGGE healthcare     ┆ 1    ┆ 1                 │
│ AOJ                  ┆ 1    ┆ 1                 │
│ FuviBaby (Việt Nam)  ┆ 1    ┆ 1                 │
│ Eucryl               ┆ 1    ┆ 1                 │
│ Hapi                 ┆ 1    ┆ 1                 │
│ Marumoto             ┆ 1    ┆ 1                 │
│ Phúc Long            ┆ 1    ┆ 1                 │
│ Kunella              ┆ 1    ┆ 1                 │
│ Delpharm Gaillard    ┆ 1    ┆ 1                 │
│ Mabio                ┆ 1    ┆ 1                 │
│ IPEK                 ┆ 1    ┆ 1                 │
│ TH True Water        ┆ 1    ┆ 1                 │
│ BINYI TOYS           ┆ 1    ┆ 1                 │
│ Bubs Full Cream      ┆ 1    ┆ 1                 │
│ Hikid Sữa Dê         ┆ 1    ┆ 1                 │
│ Nuk (Đức)            ┆ 1    ┆ 1                 │
│ Yko                  ┆ 1    ┆ 1                 │
│ AGIMEXPHARM          ┆ 1    ┆ 1                 │
│ TommeeTippee (Anh)   ┆ 1    ┆ 1                 │
│ Misa (Việt Nam)      ┆ 1    ┆ 1                 │
│ Kingphar             ┆ 1    ┆ 1                 │
│ Olympian Labs        ┆ 1    ┆ 1                 │
│ Oral-B               ┆ 1    ┆ 1                 │
│ Shimaya (Nhật Bản)   ┆ 1    ┆ 1                 │
│ Marutomo             ┆ 1    ┆ 1                 │
│ Diệp Chi Organic     ┆ 1    ┆ 1                 │
│ ARS                  ┆ 1    ┆ 1                 │
│ Dappel               ┆ 1    ┆ 1                 │
│ Morinaga Sữa nước    ┆ 1    ┆ 1                 │
│ Sudocrem             ┆ 1    ┆ 1                 │
│ Funmore              ┆ 1    ┆ 1                 │
│ NaTip                ┆ 1    ┆ 1                 │
└──────────────────────┴──────┴───────────────────┘

================================================================================
Column: manufacturer (value -> number of rows / distinct item_id count)
--------------------------------------------------------------------------------
shape: (836, 3)
┌──────────────────────────────────────────────────────────────────────┬───────┬───────────────────┐
│ manufacturer                                                         ┆ rows  ┆ distinct_item_ids │
│ ---                                                                  ┆ ---   ┆ ---               │
│ str                                                                  ┆ u32   ┆ u32               │
╞══════════════════════════════════════════════════════════════════════╪═══════╪═══════════════════╡
│ Không xác định                                                       ┆ 27465 ┆ 27465             │
│ MiDa Mec (Việt Nam )                                                 ┆ 529   ┆ 529               │
│ Mino International Co.,Ltd.                                          ┆ 82    ┆ 82                │
│ JK COLLECTION LIMITED                                                ┆ 38    ┆ 38                │
│ Unilever                                                             ┆ 28    ┆ 28                │
│ Mead Johnson Nutrition (Thailand) Ltd                                ┆ 20    ┆ 20                │
│ Công ty Cổ Phần Thực Phẩm Dinh Dưỡng Nutifood Bình Dương<br>Lô E3,   ┆ 19    ┆ 19                │
│ E4 Khu Công Nghiệp Mỹ Phước, Phường Mỹ Phước, Thành p…               ┆       ┆                   │
│ Vinamilk                                                             ┆ 17    ┆ 17                │
│ Diana Unicharm                                                       ┆ 14    ┆ 14                │
│ Hayat Việt Nam                                                       ┆ 12    ┆ 12                │
│ VitaDairy                                                            ┆ 11    ┆ 11                │
│ Công ty Bourbon Corporation<br>1-3-1 Ekimae, Kashiwazaki, Niigata,   ┆ 11    ┆ 11                │
│ 945-8611                                                             ┆       ┆                   │
│ CHI NHÁNH CÔNG TY CỔ PHẦN EVERPIA                                    ┆ 11    ┆ 11                │
│ CÔNG TY CỔ PHẦN SÀI GÒN FOOD - CHI NHÁNH VĨNH LỘC<br>Lô C13-C14/II,  ┆ 11    ┆ 11                │
│ Đường 2F, KCN Vĩnh Lộc, Xã Vĩnh Lộc A, H. Bình Chánh…                ┆       ┆                   │
│ LONG HAPPY TOYS                                                      ┆ 11    ┆ 11                │
│ Australia Deloraine Dairy Pty Ltd                                    ┆ 11    ┆ 11                │
│ Abbott Ireland, Cootehill, Co. Cavan, Ireland                        ┆ 10    ┆ 10                │
│ Công Ty Cổ Phần Sữa Việt Nam                                         ┆ 10    ┆ 10                │
│ SHANTOU CHENGHAl WENYI TOYS CO.LTD                                   ┆ 10    ┆ 10                │
│ HiPP Kft                                                             ┆ 9     ┆ 9                 │
│ SHANTOU CHENGHAIWENYI TOYS CO.LTD                                    ┆ 9     ┆ 9                 │
│ Zaza Sekerleme Gida San. Ve Tic. Ltd. Sti.<br>Fevzi Çakmak Mah.      ┆ 9     ┆ 9                 │
│ Necip Fazıl Cad. No:12 Karatay/KONYA/TURKEY                          ┆       ┆                   │
│ JUNSHI SHOES COMPANY LIMITED                                         ┆ 9     ┆ 9                 │
│ YI MIN TOYS FACTORY                                                  ┆ 9     ┆ 9                 │
│ Công ty TNHH FrieslandCampina Hà Nam<br>Cụm CN Tây Nam, TP. Phủ Lý,  ┆ 9     ┆ 9                 │
│ Tỉnh Hà Nam, Việt Nam.                                               ┆       ┆                   │
│ Nhà máy Konolfingen                                                  ┆ 9     ┆ 9                 │
│ HiPP Produktion Gmunden GmBH                                         ┆ 8     ┆ 8                 │
│ Chi nhánh công ty TNHH Quốc Tế US - Nhà máy sản xuất<br>10/8B Ấp 1 , ┆ 8     ┆ 8                 │
│ Xã Xuân Thới Thượng, Huyện Hóc Môn, Thành phố Hồ Ch…                 ┆       ┆                   │
│ THE HI CO., LTD                                                      ┆ 8     ┆ 8                 │
│ GOODFOOD CO., LTD                                                    ┆ 8     ┆ 8                 │
│ Công ty Cổ phần sữa VitaDairy Việt Nam                               ┆ 8     ┆ 8                 │
│ SHANTOU CHENGHAlWENYI TOYS CO.LTD                                    ┆ 8     ┆ 8                 │
│ Asahi Group Foods, Ltd                                               ┆ 7     ┆ 7                 │
│ P&G                                                                  ┆ 7     ┆ 7                 │
│ Oji Nepia Co., Ltd.                                                  ┆ 7     ┆ 7                 │
│ KAILIXIN TOYS CO.,LTD                                                ┆ 7     ┆ 7                 │
│ Anh Kim                                                              ┆ 6     ┆ 6                 │
│ Guangzhou Huaxi Trading Co.,Ltd                                      ┆ 6     ┆ 6                 │
│ Con Cưng                                                             ┆ 6     ┆ 6                 │
│ Pigeon                                                               ┆ 6     ┆ 6                 │
│ LEZHOU INTELLIGENT TECHNOLOGY CO.,LTD                                ┆ 6     ┆ 6                 │
│ CÔNG TY CỔ PHẦN AN AN AGRI<br>Xóm Yên Xuân , xã Diễn Phúc, huyện     ┆ 6     ┆ 6                 │
│ Diễn Châu, tỉnh Nghệ An, Việt Nam                                    ┆       ┆                   │
│ Guangdong Suntree Foodstuff Co., Ltd.                                ┆ 6     ┆ 6                 │
│ SHANTOU TONGYANG TOYS CO., LTD                                       ┆ 6     ┆ 6                 │
│ Shantou Juqi Candy Toys Industrial Co., Ltd.<br>West of Yutan Road.  ┆ 6     ┆ 6                 │
│ Longtianr. Guangyi, Chenghai Dist. Shantou City, Gua…                ┆       ┆                   │
│ XINYANG TOYS FACTORY                                                 ┆ 6     ┆ 6                 │
│ HONGKONG JASON TOYS CO., LTD                                         ┆ 6     ┆ 6                 │
│ Công ty CP dinh dưỡng Nutricare                                      ┆ 5     ┆ 5                 │
│ Nhà máy Konolfingen<br>Nestlé-Strasse 1, 3510 Konolfingen, Thụy Sĩ.  ┆ 5     ┆ 5                 │
│ Công ty TNHH MTV Thực phẩm xanh Từ Phong<br>Lô CN5, Cụm Công nghiệp  ┆ 5     ┆ 5                 │
│ Cam Thành, xã Cam Thành, huyện Cam Lộ, tỉnh Quảng Tr…                ┆       ┆                   │
│ …                                                                    ┆ …     ┆ …                 │
│ Xlear Inc. Utah 84003. USA                                           ┆ 1     ┆ 1                 │
│ BO JUN FACTORY                                                       ┆ 1     ┆ 1                 │
│ BO SHENG LONG FACTORY                                                ┆ 1     ┆ 1                 │
│ KOLMAR KOREA CO., LTD                                                ┆ 1     ┆ 1                 │
│ CÔNG TY TRÁCH NHIỆM HỮU HẠN QBB VIỆT NAM<br>Số 21 Võ Trường Toản,    ┆ 1     ┆ 1                 │
│ Phường Thảo Điền, Thành phố Thủ Đức, Thành phố Hồ Chí …              ┆       ┆                   │
│ Privatmolkerei Bauer GmbH &amp; Co. KG<br>Molkerei-Bauer-Str. 1-10   ┆ 1     ┆ 1                 │
│ 83512 Wasserburg/Inn.<br>                                            ┆       ┆                   │
│ Tempo, Vinda Paper Jiangmen City, China                              ┆ 1     ┆ 1                 │
│ RANXIAN TOYS FACTORY                                                 ┆ 1     ┆ 1                 │
│ Glico                                                                ┆ 1     ┆ 1                 │
│ 283 Đ. Hoàng Diệu, Phường 6, Quận 4, Thành phố Hồ Chí Minh           ┆ 1     ┆ 1                 │
│ MING WEI TOYS FACTORY&nbsp;                                          ┆ 1     ┆ 1                 │
│ Morinaga Milk Industry CO., LTD<br>Địa chỉ: 5-2, HigashiShimbashi    ┆ 1     ┆ 1                 │
│ 1-Chome, Minato-ku, Tokyo 105-7122, Nhật Bản<br>                     ┆       ┆                   │
│ LIAN CHUANG TOYS                                                     ┆ 1     ┆ 1                 │
│ Mino&nbsp;                                                           ┆ 1     ┆ 1                 │
│ Arla Foods amba Arinco sản xuất với sự cho phép của Meiji Co., Ltd   ┆ 1     ┆ 1                 │
│ SHANTOU CITY CHENGHAI DISTRICT JINGSHENG PLASTIC TOYS FACTORY        ┆ 1     ┆ 1                 │
│ Ariake Japan Co., Ltd<br>370-2 Kuroishi, Kosaza, Sasebo, Nagasaki,   ┆ 1     ┆ 1                 │
│ Japan                                                                ┆       ┆                   │
│ Rockit Trading Company Ltd<br>22 Irongate Road East Hasting 4175 New ┆ 1     ┆ 1                 │
│ Zealand<br>                                                          ┆       ┆                   │
│ CÔNG TY CỔ PHẦN QUỐC TẾ HOÀNG NAM<br>140 Đường số 14, KDC Him Lam,   ┆ 1     ┆ 1                 │
│ Phường Tân Hưng, Quận 7                                              ┆       ┆                   │
│ OK TOYS FACTORY                                                      ┆ 1     ┆ 1                 │
│ JIN WEI BO FACTORY                                                   ┆ 1     ┆ 1                 │
│ Artsana S.p.A, VianSaldarini Catelll, 1-22070 Grandate, Como , Ý.    ┆ 1     ┆ 1                 │
│ Nestlé Deutschland AG-Nhà máy Biessenhofen, Füssener Straße 1, 87640 ┆ 1     ┆ 1                 │
│ Biessenhofen, Đức                                                    ┆       ┆                   │
│ Anh Kim, Anh Kim, Anh Kim, Anh Kim                                   ┆ 1     ┆ 1                 │
│ SHANTOU MING DUO TOYS FACTORY                                        ┆ 1     ┆ 1                 │
│ Morinaga Milk Industry Co.,Ltd                                       ┆ 1     ┆ 1                 │
│ Shantou Juqi Candy Toys Industrial Co., Ltd. <br>West of Yutan Road. ┆ 1     ┆ 1                 │
│ Longtianr. Guangyi, Chenghai Dist. Shantou City, Gu…                 ┆       ┆                   │
│ Sung Gyung Food Co., Ltd., Factory 2<br>377 Sintanjin-ro,            ┆ 1     ┆ 1                 │
│ Daedeok-gu, Daejeon, Korea                                           ┆       ┆                   │
│ Calbee.Inc.                                                          ┆ 1     ┆ 1                 │
│ Noumi                                                                ┆ 1     ┆ 1                 │
│ NHÀ MÁY BIBICA BIÊN HÒA –&nbsp; CHI NHÁNH CÔNG TY CỔ PHẦN BIBICA     ┆ 1     ┆ 1                 │
│ <br>Khu công nghiệp Biên Hòa 1, P.An Bình, TP.Biên Hòa,…             ┆       ┆                   │
│ Marumiya Corporation<br>Saitama, Hidaka, Asahigaoka 641-1&nbsp;      ┆ 1     ┆ 1                 │
│ Gilbert                                                              ┆ 1     ┆ 1                 │
│ BOMEI TOYS FACTORY                                                   ┆ 1     ┆ 1                 │
│ SHANTOU WEIQI TECHNOLOGY INDUSTRIAL CO,LTD                           ┆ 1     ┆ 1                 │
│ Fatzbaby (Hàn Quốc)                                                  ┆ 1     ┆ 1                 │
│ PEIJIN TOYS FACTORY                                                  ┆ 1     ┆ 1                 │
│ IVENET POCHEON FOOD CO., LTD<br>4-22, Jeonggeum-ro 255beon-gil,      ┆ 1     ┆ 1                 │
│ Gasan-myeon, Pocheon-si, Gyeonggi-do, Hàn Quốc.<br>                  ┆       ┆                   │
│ AIYINGLE TOYS FACTORY                                                ┆ 1     ┆ 1                 │
│ SHANTOU CITY CHENGHAI DISTRICT BEIDIYUAN TOYS FACTORY, SHANTOU CITY  ┆ 1     ┆ 1                 │
│ CHENGHAI DISTRICT BEIDIYUAN TOYS FACTORY                             ┆       ┆                   │
│ TONG YANG FACTORY, TONG YANG FACTORY                                 ┆ 1     ┆ 1                 │
│ Shantou Juqi Candy Toys Industrial Co., Ltd.<br>West of Yutan Road.  ┆ 1     ┆ 1                 │
│ Longtianr. Guangyi, Chenghai Dist. Shantou City, Gua…                ┆       ┆                   │
│ Cetaphil                                                             ┆ 1     ┆ 1                 │
│ AO JIA FACTORY                                                       ┆ 1     ┆ 1                 │
│ Laica                                                                ┆ 1     ┆ 1                 │
│ JIALE YU TOYS&nbsp; FACTORY                                          ┆ 1     ┆ 1                 │
│ SHANTOU BAO LE CHUANG TOYS FACTORY                                   ┆ 1     ┆ 1                 │
│ CHUANGDA TOYS FACTORY                                                ┆ 1     ┆ 1                 │
│ Nhà máy sản xuất Vivera D.O.O                                        ┆ 1     ┆ 1                 │
│ Listerine                                                            ┆ 1     ┆ 1                 │
└──────────────────────────────────────────────────────────────────────┴───────┴───────────────────┘

================================================================================
Column: sale_status (value -> number of rows / distinct item_id count)
--------------------------------------------------------------------------------
shape: (2, 3)
┌─────────────┬───────┬───────────────────┐
│ sale_status ┆ rows  ┆ distinct_item_ids │
│ ---         ┆ ---   ┆ ---               │
│ str         ┆ u32   ┆ u32               │
╞═════════════╪═══════╪═══════════════════╡
│ 0           ┆ 22973 ┆ 22973             │
│ 1           ┆ 6850  ┆ 6850              │
└─────────────┴───────┴───────────────────┘

================================================================================
Column: size (value -> number of rows / distinct item_id count)
--------------------------------------------------------------------------------
shape: (97, 3)
┌───────────────────────────┬───────┬───────────────────┐
│ size                      ┆ rows  ┆ distinct_item_ids │
│ ---                       ┆ ---   ┆ ---               │
│ str                       ┆ u32   ┆ u32               │
╞═══════════════════════════╪═══════╪═══════════════════╡
│ Không xác định            ┆ 27880 ┆ 27880             │
│ 0-3M                      ┆ 255   ┆ 255               │
│ 6-9M                      ┆ 209   ┆ 209               │
│ 1Y                        ┆ 201   ┆ 201               │
│ 9 tháng                   ┆ 111   ┆ 111               │
│ 13                        ┆ 93    ┆ 93                │
│ NB                        ┆ 91    ┆ 91                │
│ 1-2Y                      ┆ 70    ┆ 70                │
│ 3-6M                      ┆ 54    ┆ 54                │
│ 14                        ┆ 52    ┆ 52                │
│ 6 tháng                   ┆ 50    ┆ 50                │
│ 9-12M                     ┆ 49    ┆ 49                │
│ 3 tháng                   ┆ 46    ┆ 46                │
│ 2T                        ┆ 45    ┆ 45                │
│ 12                        ┆ 38    ┆ 38                │
│ 2Y                        ┆ 37    ┆ 37                │
│ 9M                        ┆ 29    ┆ 29                │
│ 18-24M                    ┆ 28    ┆ 28                │
│ 12-18M                    ┆ 25    ┆ 25                │
│ 7-8 tuổi                  ┆ 22    ┆ 22                │
│ 3Y                        ┆ 19    ┆ 19                │
│ 21                        ┆ 19    ┆ 19                │
│ S                         ┆ 19    ┆ 19                │
│ 7-8Y                      ┆ 18    ┆ 18                │
│ 4Y                        ┆ 17    ┆ 17                │
│ 9                         ┆ 17    ┆ 17                │
│ 12 tháng                  ┆ 16    ┆ 16                │
│ 19                        ┆ 16    ┆ 16                │
│ 2-3Y                      ┆ 16    ┆ 16                │
│ 0 tháng                   ┆ 16    ┆ 16                │
│ 3T                        ┆ 16    ┆ 16                │
│ S17                       ┆ 14    ┆ 14                │
│ 17                        ┆ 13    ┆ 13                │
│ 23                        ┆ 13    ┆ 13                │
│ 15                        ┆ 13    ┆ 13                │
│ 0-6M                      ┆ 12    ┆ 12                │
│ 1-2 tuổi                  ┆ 12    ┆ 12                │
│ 24                        ┆ 10    ┆ 10                │
│ 16                        ┆ 10    ┆ 10                │
│ L                         ┆ 10    ┆ 10                │
│ 1T                        ┆ 9     ┆ 9                 │
│ 1                         ┆ 9     ┆ 9                 │
│ Sơ sinh                   ┆ 8     ┆ 8                 │
│ 24 tháng                  ┆ 8     ┆ 8                 │
│ 6T                        ┆ 7     ┆ 7                 │
│ 2-3T                      ┆ 7     ┆ 7                 │
│ M                         ┆ 6     ┆ 6                 │
│ 0-1Y                      ┆ 6     ┆ 6                 │
│ 2                         ┆ 5     ┆ 5                 │
│ 5T                        ┆ 5     ┆ 5                 │
│ 18 tháng                  ┆ 4     ┆ 4                 │
│ SS                        ┆ 4     ┆ 4                 │
│ 5-6Y                      ┆ 4     ┆ 4                 │
│ 10                        ┆ 3     ┆ 3                 │
│ B85                       ┆ 3     ┆ 3                 │
│ 3                         ┆ 3     ┆ 3                 │
│ 3-4Y                      ┆ 3     ┆ 3                 │
│ 4-14 tháng                ┆ 2     ┆ 2                 │
│ 0-3 tuổi                  ┆ 2     ┆ 2                 │
│ 4-5T                      ┆ 2     ┆ 2                 │
│ 1Y, 1Y                    ┆ 2     ┆ 2                 │
│ XXL                       ┆ 2     ┆ 2                 │
│ NB (Dưới 5kg)             ┆ 2     ┆ 2                 │
│ 12-24M                    ┆ 2     ┆ 2                 │
│ XL                        ┆ 2     ┆ 2                 │
│ 24*28cm                   ┆ 1     ┆ 1                 │
│ 34                        ┆ 1     ┆ 1                 │
│ 24-36 tháng               ┆ 1     ┆ 1                 │
│ Từ 7 tháng                ┆ 1     ┆ 1                 │
│ 22                        ┆ 1     ┆ 1                 │
│ 12-36 tháng               ┆ 1     ┆ 1                 │
│ 4-6 tháng                 ┆ 1     ┆ 1                 │
│ XXL ( >15kg) - 48+6 miếng ┆ 1     ┆ 1                 │
│ 25x28cm                   ┆ 1     ┆ 1                 │
│ 36                        ┆ 1     ┆ 1                 │
│ 110                       ┆ 1     ┆ 1                 │
│ 28*36cm                   ┆ 1     ┆ 1                 │
│ M(7-12kg) - 76 miếng      ┆ 1     ┆ 1                 │
│ 4Y, 4Y, 4Y                ┆ 1     ┆ 1                 │
│ 20                        ┆ 1     ┆ 1                 │
│ 11                        ┆ 1     ┆ 1                 │
│ 90*140cm                  ┆ 1     ┆ 1                 │
│ Từ 2 tuổi                 ┆ 1     ┆ 1                 │
│ XXL (>15kg) 54 miếng      ┆ 1     ┆ 1                 │
│ L(9-14kg) - 68 miếng      ┆ 1     ┆ 1                 │
│ 4-5Y                      ┆ 1     ┆ 1                 │
│ B75                       ┆ 1     ┆ 1                 │
│ 1-3 tháng                 ┆ 1     ┆ 1                 │
│ M(6-11kg) - 64 miếng      ┆ 1     ┆ 1                 │
│ XXL(15-25kg) - 26 miếng   ┆ 1     ┆ 1                 │
│ 0-36 tháng                ┆ 1     ┆ 1                 │
│ 2Y, 1Y                    ┆ 1     ┆ 1                 │
│ 4T                        ┆ 1     ┆ 1                 │
│ S (4-8kg) - 80 miếng      ┆ 1     ┆ 1                 │
│ L (9-14kg) - 72 miếng     ┆ 1     ┆ 1                 │
│ 60*100cm                  ┆ 1     ┆ 1                 │
│ 26                        ┆ 1     ┆ 1                 │
└───────────────────────────┴───────┴───────────────────┘
```

```python
from datetime import date, timedelta
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

txn_df = data_frames["transaction_full_2025.parquet"]

sales = (
    txn_df.select(
        pl.col("updated_date").cast(pl.Datetime, strict=False).alias("date"),
        pl.col("quantity").fill_null(0).alias("quantity"),
    )
    .drop_nulls("date")
    .with_columns(pl.col("date").dt.date().alias("day"))
)

max_day = sales.select(pl.col("day").max()).item()
month_end = date(max_day.year, max_day.month, 1)
month_start = date((month_end.year * 12 + month_end.month - 12) // 12, ((month_end.year * 12 + month_end.month - 12) % 12) + 1, 1)
day_start = max_day - timedelta(days=364)

monthly_raw = (
    sales.with_columns(pl.col("date").dt.strftime("%Y-%m").alias("month"))
    .group_by("month")
    .agg(pl.col("quantity").sum().alias("sale_quantity"))
    .sort("month")
)

daily_raw = (
    sales.filter(pl.col("day") >= pl.lit(day_start))
    .group_by("day")
    .agg(pl.col("quantity").sum().alias("sale_quantity"))
    .sort("day")
)

monthly_index = pl.DataFrame({
    "month": [d.strftime("%Y-%m") for d in pl.date_range(month_start, month_end, interval="1mo", eager=True).to_list()]
})

monthly = (
    monthly_index.join(monthly_raw, on="month", how="left")
    .with_columns(pl.col("sale_quantity").fill_null(0))
    .sort("month")
)

daily = (
    pl.DataFrame({"day": pl.date_range(day_start, max_day, interval="1d", eager=True)})
    .join(daily_raw, on="day", how="left")
    .with_columns(pl.col("sale_quantity").fill_null(0))
    .sort("day")
)

print("Transaction file: transaction_full_2025.parquet")
print("Date column: updated_date")
print("Quantity column: quantity")

print("\nMonthly sale quantity (12 months)")
print(monthly)

print("\nDaily sale quantity (365 days)")
print(daily)

fig, axes = plt.subplots(2, 1, figsize=(14, 8), constrained_layout=True)

axes[0].bar(monthly["month"].to_list(), monthly["sale_quantity"].to_list(), color="#2F6BFF", width=0.6)
axes[0].set_title("Sale Quantity by Month")
axes[0].set_ylabel("Quantity")
axes[0].tick_params(axis="x", rotation=45)

axes[1].plot(daily["day"].to_list(), daily["sale_quantity"].to_list(), color="#E35D2F", linewidth=1)
axes[1].set_title("Sale Quantity by Day")
axes[1].set_ylabel("Quantity")
axes[1].xaxis.set_major_locator(mdates.MonthLocator())
axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
axes[1].tick_params(axis="x", rotation=45)

plt.show()
```

```text
Transaction file: transaction_full_2025.parquet
Date column: updated_date
Quantity column: quantity

Monthly sale quantity (12 months)
shape: (12, 2)
┌─────────┬───────────────┐
│ month   ┆ sale_quantity │
│ ---     ┆ ---           │
│ str     ┆ i32           │
╞═════════╪═══════════════╡
│ 2025-01 ┆ 5111048       │
│ 2025-02 ┆ 4496932       │
│ 2025-03 ┆ 4822234       │
│ 2025-04 ┆ 4902122       │
│ 2025-05 ┆ 5451526       │
│ 2025-06 ┆ 5366211       │
│ 2025-07 ┆ 5268305       │
│ 2025-08 ┆ 5875812       │
│ 2025-09 ┆ 5665199       │
│ 2025-10 ┆ 6050888       │
│ 2025-11 ┆ 6639636       │
│ 2025-12 ┆ 178556        │
└─────────┴───────────────┘

Daily sale quantity (365 days)
shape: (365, 2)
┌────────────┬───────────────┐
│ day        ┆ sale_quantity │
│ ---        ┆ ---           │
│ date       ┆ i32           │
╞════════════╪═══════════════╡
│ 2024-12-02 ┆ 0             │
│ 2024-12-03 ┆ 0             │
│ 2024-12-04 ┆ 0             │
│ 2024-12-05 ┆ 0             │
│ 2024-12-06 ┆ 0             │
│ 2024-12-07 ┆ 0             │
│ 2024-12-08 ┆ 0             │
│ 2024-12-09 ┆ 0             │
│ 2024-12-10 ┆ 0             │
│ 2024-12-11 ┆ 0             │
│ 2024-12-12 ┆ 0             │
│ 2024-12-13 ┆ 0             │
│ 2024-12-14 ┆ 0             │
│ 2024-12-15 ┆ 0             │
│ 2024-12-16 ┆ 0             │
│ 2024-12-17 ┆ 0             │
│ 2024-12-18 ┆ 0             │
│ 2024-12-19 ┆ 0             │
│ 2024-12-20 ┆ 0             │
│ 2024-12-21 ┆ 0             │
│ 2024-12-22 ┆ 0             │
│ 2024-12-23 ┆ 0             │
│ 2024-12-24 ┆ 0             │
│ 2024-12-25 ┆ 0             │
│ 2024-12-26 ┆ 0             │
│ 2024-12-27 ┆ 0             │
│ 2024-12-28 ┆ 0             │
│ 2024-12-29 ┆ 0             │
│ 2024-12-30 ┆ 0             │
│ 2024-12-31 ┆ 0             │
│ 2025-01-01 ┆ 185001        │
│ 2025-01-02 ┆ 159855        │
│ 2025-01-03 ┆ 145932        │
│ 2025-01-04 ┆ 152668        │
│ 2025-01-05 ┆ 165224        │
│ 2025-01-06 ┆ 161048        │
│ 2025-01-07 ┆ 151772        │
│ 2025-01-08 ┆ 148278        │
│ 2025-01-09 ┆ 147081        │
│ 2025-01-10 ┆ 165719        │
│ 2025-01-11 ┆ 168929        │
│ 2025-01-12 ┆ 188450        │
│ 2025-01-13 ┆ 164189        │
│ 2025-01-14 ┆ 145658        │
│ 2025-01-15 ┆ 172109        │
│ 2025-01-16 ┆ 166982        │
│ 2025-01-17 ┆ 160029        │
│ 2025-01-18 ┆ 172668        │
│ 2025-01-19 ┆ 197398        │
│ 2025-01-20 ┆ 179740        │
│ …          ┆ …             │
│ 2025-10-13 ┆ 192648        │
│ 2025-10-14 ┆ 180914        │
│ 2025-10-15 ┆ 238431        │
│ 2025-10-16 ┆ 191386        │
│ 2025-10-17 ┆ 176459        │
│ 2025-10-18 ┆ 186207        │
│ 2025-10-19 ┆ 197745        │
│ 2025-10-20 ┆ 169707        │
│ 2025-10-21 ┆ 189401        │
│ 2025-10-22 ┆ 169482        │
│ 2025-10-23 ┆ 166336        │
│ 2025-10-24 ┆ 191333        │
│ 2025-10-25 ┆ 249412        │
│ 2025-10-26 ┆ 268494        │
│ 2025-10-27 ┆ 0             │
│ 2025-10-28 ┆ 164983        │
│ 2025-10-29 ┆ 167921        │
│ 2025-10-30 ┆ 165736        │
│ 2025-10-31 ┆ 205154        │
│ 2025-11-01 ┆ 219717        │
│ 2025-11-02 ┆ 261426        │
│ 2025-11-03 ┆ 209106        │
│ 2025-11-04 ┆ 208079        │
│ 2025-11-05 ┆ 213380        │
│ 2025-11-06 ┆ 202267        │
│ 2025-11-07 ┆ 170537        │
│ 2025-11-08 ┆ 206892        │
│ 2025-11-09 ┆ 232778        │
│ 2025-11-10 ┆ 225310        │
│ 2025-11-11 ┆ 289447        │
│ 2025-11-12 ┆ 235077        │
│ 2025-11-13 ┆ 195947        │
│ 2025-11-14 ┆ 189195        │
│ 2025-11-15 ┆ 222111        │
│ 2025-11-16 ┆ 217695        │
│ 2025-11-17 ┆ 198260        │
│ 2025-11-18 ┆ 175673        │
│ 2025-11-19 ┆ 172987        │
│ 2025-11-20 ┆ 185886        │
│ 2025-11-21 ┆ 193216        │
│ 2025-11-22 ┆ 290476        │
│ 2025-11-23 ┆ 288792        │
│ 2025-11-24 ┆ 229810        │
│ 2025-11-25 ┆ 256124        │
│ 2025-11-26 ┆ 227386        │
│ 2025-11-27 ┆ 205597        │
│ 2025-11-28 ┆ 227150        │
│ 2025-11-29 ┆ 231346        │
│ 2025-11-30 ┆ 257969        │
│ 2025-12-01 ┆ 178556        │
└────────────┴───────────────┘
```

![output image 13-1](images/cell-13-1.png)

```python

```

