import pandas as pd
import os

# Cấu hình đường dẫn
input_file = 'dataset/network/kpi_15_mins_3_months.csv'
output_dir = 'dataset/network/'

# Tạo thư mục output nếu chưa có
os.makedirs(output_dir, exist_ok=True)

# Đọc file CSV
df = pd.read_csv(input_file)

# Đổi tên timestamp thành date
if 'timestamp' in df.columns:
    df.rename(columns={'timestamp': 'date'}, inplace=True)

# Xác định tên cột enodeB trong file gốc (đề phòng viết hoa/thường)
# Kiểm tra xem là 'enodeb' hay 'enodeB'
enodeb_col_name = 'enodeb' if 'enodeb' in df.columns else 'enodeB'

# Lấy danh sách các cell unique
cell_names = df['cell_name'].unique()

# Xử lý từng cell
for cell_name in cell_names:
    # 1. Lọc dữ liệu của cell hiện tại
    cell_df = df[df['cell_name'] == cell_name].copy()

    # Sắp xếp theo thời gian
    cell_df.sort_values(by='date', inplace=True)

    # 2. Lấy tên enodeB (Lấy giá trị dòng đầu tiên vì cùng 1 cell thì cùng enodeB)
    # Chuyển về string để tránh lỗi nếu là số
    current_enodeb = "Unknown"
    if enodeb_col_name in cell_df.columns:
        current_enodeb = str(cell_df[enodeb_col_name].iloc[0])

    # 3. Tạo tên file theo định dạng: enodeB_cellname.csv
    file_name = f"{current_enodeb}_{cell_name}.csv"
    output_file = os.path.join(output_dir, file_name)

    # 4. Xóa cột cell_name và enodeb (Sau khi đã lấy tên để đặt cho file)
    cols_to_drop = ['cell_name', 'enodeb', 'enodeB']
    cell_df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    # 5. Lưu file
    cell_df.to_csv(output_file, index=False)

    print(f'Processed: {cell_name} -> Saved as: {file_name}')

print('All files processed successfully.')