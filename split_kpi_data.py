import pandas as pd
import os
from tqdm import tqdm # Thư viện tạo thanh tiến trình (nếu chưa có pip install tqdm)

# --- CẤU HÌNH ---
input_file = 'dataset/kpi/kpi_data.csv'   
output_dir = 'dataset/kpi_processed/'     
os.makedirs(output_dir, exist_ok=True)

# Các cột dữ liệu cần giữ lại
cols_to_keep = ['date', 'ps_traffic_mb', 'avg_rrc_connected_user', 
                'prb_dl_used', 'prb_dl_available_total', 'prb_utilization']
# ----------------

print(f"🔄 Đang đọc dữ liệu từ {input_file}...")

# 1. Đọc file (Xử lý lỗi BOM nếu có)
try:
    df = pd.read_csv(input_file, encoding='utf-8-sig')
except:
    df = pd.read_csv(input_file)

# 2. Chuẩn hóa tên cột (Global Cleaning)
df.columns = df.columns.str.strip().str.lower()
print(f"📋 Các cột tìm thấy: {df.columns.tolist()}")

# Đổi tên cột 'data' thành 'date' nếu tồn tại
if 'data' in df.columns:
    print("⚠️ Đổi tên cột 'data' -> 'date'")
    df.rename(columns={'data': 'date'}, inplace=True)

# Kiểm tra các cột bắt buộc
if 'date' not in df.columns:
    raise KeyError("❌ Không tìm thấy cột 'date' (hoặc 'data').")
if 'cell_name' not in df.columns:
    raise KeyError("❌ Không tìm thấy cột 'cell_name'.")

# Kiểm tra thiếu các cột chỉ số KPI
missing_cols = [c for c in cols_to_keep if c not in df.columns]
if missing_cols:
    raise KeyError(f"❌ File gốc thiếu các cột số liệu: {missing_cols}")

# Chuyển đổi định dạng thời gian và chuẩn hóa tên cell
df['date'] = pd.to_datetime(df['date'])
df['cell_name'] = df['cell_name'].astype(str).str.strip()

# 3. Xử lý tách file theo từng Cell
unique_cells = df['cell_name'].unique()
print(f"✅ Tìm thấy {len(unique_cells)} cells khác nhau. Bắt đầu tách file...")

# Sử dụng groupby để gom nhóm dữ liệu theo cell_name (Hiệu năng cao hơn for loop thường)
count_success = 0

# Tqdm giúp hiển thị thanh % tiến trình
for cell_name, df_cell in tqdm(df.groupby('cell_name'), total=len(unique_cells)):
    try:
        # Nếu tên cell rỗng hoặc nan thì bỏ qua
        if not cell_name or str(cell_name).lower() == 'nan':
            continue

        # Chỉ lấy các cột cần thiết
        df_cell = df_cell[cols_to_keep].copy()

        # 4. Resample (Lấp đầy khoảng trống thời gian cho từng cell)
        df_cell = df_cell.sort_values('date').set_index('date')
        
        # Resample 15 phút, điền 0 vào chỗ thiếu
        df_cell = df_cell.resample('15T').mean().fillna(0)
        df_cell = df_cell.reset_index()

        # 5. Lưu file
        # Làm sạch tên file (tránh các ký tự đặc biệt gây lỗi hệ thống file)
        safe_filename = "".join([c for c in cell_name if c.isalnum() or c in (' ', '-', '_')]).strip()
        output_file = os.path.join(output_dir, f'{safe_filename}.csv')
        
        df_cell.to_csv(output_file, index=False)
        count_success += 1
        
    except Exception as e:
        print(f"❌ Lỗi khi xử lý cell '{cell_name}': {e}")

print(f"\n🎉 HOÀN TẤT! Đã lưu thành công {count_success}/{len(unique_cells)} files vào thư mục '{output_dir}'.")