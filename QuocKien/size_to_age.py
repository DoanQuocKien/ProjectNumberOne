import re

def standardize_age(text):
    raw_text = str(text).strip()
    clean_text = raw_text.lower()
    
    # 1. Nhóm Kích thước (Thường là phụ kiện: khăn, lót, chiếu)
    # Quy về 0.5Y (trung bình tuổi dùng nhiều nhất) hoặc bạn có thể sửa thành "All"
    if re.search(r'(\*|x\d|cm)', clean_text):
        return 0.5

    # 2. Nhóm Đồ cho mẹ (Size áo lót bầu/sau sinh)
    if re.search(r'\bb\d{2}\b', clean_text):
        return "Adult"

    # 3. Nhóm Size giày/chiều cao đặc biệt
    if 's17' in clean_text: return 1.0
    if '110' in clean_text: return 5.0

    # 4. Các trường hợp không xác định rõ ràng
    if "không xác định" in clean_text or not clean_text:
        return "None"

    # 5. Xử lý size tã / Quần áo chuẩn (S, M, L, XL...)
    diaper_map = {
        r'\bnb\b': 0, r'\bss\b': 0, r'\bsơ sinh\b': 0,
        r'\bs\b': 0.25, r'\bm\b': 0.6, r'\bl\b': 1.2,
        r'\bxl\b': 2.0, r'\bxxl\b': 3.5
    }
    for pattern, val in diaper_map.items():
        if re.search(pattern, clean_text): return val

    # 6. Xử lý khoảng (VD: 0-3M, 1-2 tuổi, 18-24M)
    range_match = re.search(r'(\d+\.?\d*)\s*-\s*(\d+\.?\d*)', clean_text)
    if range_match:
        s, e = float(range_match.group(1)), float(range_match.group(2))
        avg = (s + e) / 2
        if any(x in clean_text for x in ['m', 'tháng']):
            return round(avg / 12, 3)
        return avg

    # 7. Xử lý số đơn lẻ kèm đơn vị
    m_match = re.search(r'(\d+\.?\d*)\s*(m|tháng)', clean_text)
    if m_match: return round(float(m_match.group(1)) / 12, 3)
    
    y_match = re.search(r'(\d+\.?\d*)\s*(y|t|tuổi)', clean_text)
    if y_match: return float(y_match.group(1))

    # 8. Xử lý số thuần túy (VD: 9, 12, 2, 3)
    # Logic: > 6 là tháng, <= 6 là tuổi
    pure_num = re.search(r'^(\d+)$', clean_text)
    if pure_num:
        val = float(pure_num.group(1))
        if val > 6:
            return round(val/12, 3)
        else:
            return val

    return "Check tay"

# --- CHẠY THỬ VỚI DỮ LIỆU CỦA BẠN ---

# Paste nội dung bảng vào đây
raw_data = """
Không xác định                 | Rác / Cần check     
0-3M                           | 0.125
6-9M                           | 0.625
1Y                             | 1.0
9 tháng                        | 0.75
13                             | 1.083
NB                             | 0
1-2Y                           | 1.5
3-6M                           | 0.375
14                             | 1.167
6 tháng                        | 0.5
9-12M                          | 0.875
3 tháng                        | 0.25
2T                             | 2.0
12                             | 1.0
2Y                             | 2.0
9M                             | 0.75
18-24M                         | 1.75
12-18M                         | 1.25
7-8 tuổi                       | 7.5
3Y                             | 3.0
21                             | 1.75
S                              | 0.25
7-8Y                           | 7.5
4Y                             | 4.0
9                              | 0.75
12 tháng                       | 1.0
19                             | 1.583
2-3Y                           | 2.5
0 tháng                        | 0.0
3T                             | 3.0
S17                            | Rác / Cần check
17                             | 1.417
23                             | 1.917
15                             | 1.25
0-6M                           | 0.25
1-2 tuổi                       | 1.5
24                             | 2.0
16                             | 1.333
L                              | 1.2
1T                             | 1.0
1                              | 1.0
Sơ sinh                        | 0
24 tháng                       | 2.0
6T                             | 6.0
2-3T                           | 2.5
M                              | 0.6
0-1Y                           | 0.5
2                              | 2.0
5T                             | 5.0
18 tháng                       | 1.5
SS                             | 0
5-6Y                           | 5.5
10                             | 0.833
B85                            | Rác / Cần check
3                              | 3.0
3-4Y                           | 3.5
4-14 tháng                     | 0.75
0-3 tuổi                       | 1.5
4-5T                           | 4.5
1Y, 1Y                         | 1.0
XXL                            | 3.5
NB (Dưới 5kg)                  | 0
12-24M                         | 1.5
XL                             | 2.0
24*28cm                        | Rác / Cần check
34                             | 2.833
24-36 tháng                    | 2.5
Từ 7 tháng                     | 0.583
22                             | 1.833
12-36 tháng                    | 2.0
4-6 tháng                      | 0.417
XXL ( >15kg) - 48+6 miếng      | 3.5
25x28cm                        | Rác / Cần check
36                             | 3.0
110                            | 9.167
28*36cm                        | Rác / Cần check
M(7-12kg) - 76 miếng           | 0.6
4Y, 4Y, 4Y                     | 4.0
20                             | 1.667
11                             | 0.917
90*140cm                       | Rác / Cần check
Từ 2 tuổi                      | 2.0
XXL (>15kg) 54 miếng           | 3.5
L(9-14kg) - 68 miếng           | 1.2
4-5Y                           | 4.5
B75                            | Rác / Cần check
1-3 tháng                      | 0.167
M(6-11kg) - 64 miếng           | 0.6
XXL(15-25kg) - 26 miếng        | 3.5
0-36 tháng                     | 1.5
2Y, 1Y                         | 2.0
4T                             | 4.0
S (4-8kg) - 80 miếng           | 0.25
L (9-14kg) - 72 miếng          | 1.2
60*100cm                       | Rác / Cần check
26                             | 2.167
""" 

print(f"{'GIÁ TRỊ GỐC':<25} | {'QUY ĐỔI (Y)':<15}")
print("-" * 45)

for line in raw_data.strip().split('\n'):
    clean_line = line.replace('│', '').strip()
    if not clean_line or '---' in clean_line or 'size' in clean_line: continue
    
    first_col = re.split(r'[┆|│]', clean_line)[0].strip()
    if first_col:
        print(f"{first_col:<25} | {standardize_age(first_col):<15}")