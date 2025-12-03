import pandas as pd
import os

# --- CẤU HÌNH ---
input_file = 'dataset/kpi/kpi_data.csv'   
target_cell = 'EnodebF121' # <--- Đã sửa chữ 'E' viết hoa cho khớp với dữ liệu của bạn
output_dir = 'dataset/kpi_processed/'     
os.makedirs(output_dir, exist_ok=True)
# ----------------

print(f"🔄 Đang đọc dữ liệu từ {input_file}...")

# 1. Đọc file (Xử lý lỗi BOM nếu có)
try:
    df = pd.read_csv(input_file, encoding='utf-8-sig')
except:
    df = pd.read_csv(input_file)

# 2. Chuẩn hóa tên cột (Xử lý vấn đề 'data' vs 'date')
# Xóa khoảng trắng thừa (ví dụ " data")
df.columns = df.columns.str.strip()
# Chuyển hết về chữ thường
df.columns = df.columns.str.lower()

print(f"📋 Các cột tìm thấy: {df.columns.tolist()}")

# SỬA LỖI CHÍNH: Đổi tên cột 'data' thành 'date' nếu nó tồn tại
if 'data' in df.columns:
    print("⚠️ Phát hiện cột tên là 'data', đang đổi tên thành 'date'...")
    df.rename(columns={'data': 'date'}, inplace=True)

# Kiểm tra lại lần cuối
if 'date' not in df.columns:
    raise KeyError(f"❌ Vẫn không tìm thấy cột thời gian. Hãy kiểm tra lại header của file CSV.")

# Chuyển đổi sang định dạng thời gian
df['date'] = pd.to_datetime(df['date'])

# 3. Lọc dữ liệu của cell đích
print(f"🔍 Đang lọc dữ liệu cho cell: {target_cell}...")

# Kiểm tra cột cell_name
if 'cell_name' not in df.columns:
     raise KeyError(f"❌ Không tìm thấy cột 'cell_name'.")

# Lưu ý: trim khoảng trắng ở dữ liệu cell_name để tránh lỗi "EnodebF121 "
df['cell_name'] = df['cell_name'].str.strip()

df_cell = df[df['cell_name'] == target_cell].copy()

if df_cell.empty:
    print(f"❌ CẢNH BÁO: Không tìm thấy dòng dữ liệu nào cho '{target_cell}'.")
    print(f"   Các cell có trong file là (5 cái đầu): {df['cell_name'].unique()[:5]}")
else:
    # 4. Chọn các cột KPI
    cols_to_keep = ['date', 'ps_traffic_mb', 'avg_rrc_connected_user', 
                    'prb_dl_used', 'prb_dl_available_total', 'prb_utilization']
    
    # Kiểm tra thiếu cột
    missing = [c for c in cols_to_keep if c not in df_cell.columns]
    if missing:
        raise KeyError(f"❌ Thiếu các cột số liệu: {missing}")

    df_cell = df_cell[cols_to_keep]

    # 5. Resample (Lấp đầy khoảng trống thời gian)
    df_cell = df_cell.sort_values('date').set_index('date')
    # Dùng phương pháp 15 phút
    df_cell = df_cell.resample('15T').mean().fillna(0)
    df_cell = df_cell.reset_index()

    # 6. Lưu file
    output_file = os.path.join(output_dir, f'{target_cell}.csv')
    df_cell.to_csv(output_file, index=False)

    print(f"✅ THÀNH CÔNG! File đã lưu tại: {output_file}")
    print(f"📊 Kích thước dữ liệu: {df_cell.shape}")
    print(df_cell.head())